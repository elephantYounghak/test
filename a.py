import multiprocessing
import os
import threading
import time
import numpy as np
import copy
from collections import Counter
from itertools import chain

from ultralytics import YOLO

from lib.init import *
from lib.function import *

from lib.mysql import db_connect
from lib.redis import redis_connect
from lib.flask import BroadcastServer
from lib.tracker import *

from lib.grabber import StreamGrabber
# from lib.darknet import DarknetWrapper
from lib.yolox_model import Yolox

from broadcast import get_broadcast_display

import cv2
import queue


# import ptvsd

# # Allow other computers to attach to ptvsd at this IP address and port.
# ptvsd.enable_attach(address=('0.0.0.0', 5678), redirect_output=True)

# # Pause the program until a remote debugger is attached
# ptvsd.wait_for_attach()


class VideoWriter(threading.Thread):
    def __init__(self, filename, fps, frame_size):
        threading.Thread.__init__(self)
        self.filename = filename
        self.fps = fps
        self.frame_size = frame_size
        self.frame_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.writer = None

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)
        
        while not self.stop_flag.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                self.writer.write(frame)
            except queue.Empty:
                continue

        # Write any remaining frames
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.writer.write(frame)

        self.writer.release()
        print("record done")
        time.sleep(0.001)

    def add_frame(self, frame):
        self.frame_queue.put(frame)

    def stop(self):
        self.stop_flag.set()
        self.join()


class BomWorker():
    def __init__(self, data):
        self.is_stopped = False
        self.init_cctv(data)

        self.clear()

    def clear(self):
        #Process
        self.grabber = None

        self.fire_idxes = []
        self.frame_list = []

        self.debug_time = time.time()
        self.notify_time = time.time()
        # self.is_output_debug = False

        #Meta, Trackers
        self.tracker = ObjectTracker()
        self.results = []
        self.model_results = {}
        self.meta = {}
        self.event_meta = {}

        self.layer_list = []
        self.count_layers = []
        self.hit_layers = []
        self.accrue_layers = []
        self.map_layers = []
        self.is_map_bind = {}

        #Frame, State
        self.state = 'disconnect'
        self.is_play = False
        self.fps = 0
        self.frame = None
        self.frame_cnt = 0
        self.frame_w, self.frame_h = [0, 0]
        self.prev_frame = None
        self.prev_frame_cnt = 0
        self.dt_update = time.time()

        self.read_idxes = []
        self.detect_idxes = []
        self.next_frames = deque()
        self.prev_frames = deque()

        #Video writer
        # self.write_frame = deque()
        self.is_detect = False
        self.is_writing = False
        self.write_frames = []
        self.writer_time = 0
        self.write_filename = None

        db['main'].set_cctv_state(self.cctv_id, 'disconnect', self.get_preview_url(), system.get_port())

        self.disp_frame = None
        self.recording = False
        self.tmp_buffer = [] # frame buffer for event video
        self.dt_broadcast = time.time()
        self.evt_vid_buffer = [] # list of frame buffers for writing an event video
        self.jpg_filepath = None
        self.mp4_filepath = None
        self.last_detect_time = None
        self.start_record = False
        self.lines = None
        self.recording_frame_cnt = 0
        self.video_writer = None
        self.detection_time = None
        
        self.object_count = -1

    def init_cctv(self, data):
        self.cctv_id = data['cctv_id']
        self.cctv_name = data['name']
        self.purpose = data['purpose']
        self.cctv_enable = data['cctv_enable']
        self.obj_height = data['obj_height']
        self.duration = data['duration']

        self.fire_confidence = data['fire_confidence'] if data['fire_confidence'] > 0 else 0.1
        self.smoke_confidence = data['smoke_confidence'] if data['smoke_confidence'] > 0 else 0.1
        self.train_confidence = data['train_confidence'] if data['train_confidence'] > 0 else 0.1

        self.layer_dic = self.init_roi(pts=[data['roi1'], data['roi2']])

        self.interval = data['interval']
        self.http_url = data['http_url']
        self.snmp = {
            'ip': data['snmp_ip'],
            'port': data['snmp_port'],
            'oid': data['snmp_oid'],
        }
        self.url = db['main'].conv_url(data)
        # if self.protocol == 'file':
        #     db['main'].set_delete_count(self.cctv_id)
        # self.url = self.url.replace('amp;', '')
        # self.url = self.url.replace('!', '%21')
        # self.protocol
        self.is_work = True

        # detector.cctv_id = data['cctv_id']

    def init_roi(self, pts):
        roi_dic = {}
        for i, pt in enumerate(pts):
            layer_loc = []
            try:
                pt = eval(pt) if len(pt) > 4 else []
                if len(pt) == 4:
                    pt = sorted(pt, key=lambda x:x[1])

                    up_pts = sorted(pt[:2], key=lambda x:x[0])
                    down_pts = sorted(pt[2:], key=lambda x:x[0])

                    LT, RT, LB, RB = up_pts[0], up_pts[1], down_pts[0], down_pts[1]
                    layer_loc = [LT, LB, RB, RT]
            except Exception as ex:
                log.error('CCTV "{}" >> roi{} setting error'.format(self.cctv_name, i+1))
                pass
            roi_dic['roi{}'.format(i+1)] = layer_loc
        return roi_dic

    def connect(self):
        if self.grabber is not None:
            print('del grabber')
            self.grabber.proc_terminate()
            #time.sleep(1)
            del self.grabber
            self.grabber = None

        self.frame_cnt = 0
        self.is_stopped = False

        if self.cctv_enable == 'enable':
            print('del grabber')
            log.debug('{} >> {} connecting.'.format(self.cctv_name, self.url))

            self.grabber = StreamGrabber(
                url=self.url,
                on_state=self.on_state,
                on_frame=self.on_frame,
            )
            self.grabber.daemon = True
            self.grabber.start()

    def close(self):
        self.is_play = False
        self.is_stopped = True
        if self.grabber is not None:
            print('closed grabber')
            self.grabber.proc_terminate()
            #time.sleep(1)
            #del self.grabber
            self.grabber = None
            self.clear()

        if self.video_writer:
            self.video_writer.stop()

    def on_state(self, is_play):
        if self.is_stopped:
            return

        import traceback

        try:
            if is_play:
                self.fps = self.grabber.fps
                self.frame_w, self.frame_h = self.grabber.frame_w, self.grabber.frame_h
                self.read_idxes = np.arange(1, self.fps+1)
                if self.fps <= FRAME_READ_CNT:
                    self.read_idxes = conv_spaced_list(self.read_idxes, FRAME_READ_CNT, is_end=False)

                self.detect_idxes = dict([(int(i), []) for i in np.arange(1, self.fps+1)])
                for model, option in list(DETECTION_OPTIONS.items()):
                    if 'spaced' in option:
                        for i in conv_spaced_list(self.read_idxes, option['spaced'], is_end=False):
                            if model not in self.detect_idxes[int(i)]:
                                self.detect_idxes[int(i)].append(model)
                        self.model_results[model] = []
                counts = Counter(chain.from_iterable(self.detect_idxes.values()))
                # detector.fps = counts
                detector.fps = self.fps
                self.grabber.read_idxes = self.read_idxes
                self.grabber.detect_idxes = self.detect_idxes
                #self.frame_w, self.frame_h = self.grabber.width, self.grabber.height
                self.fire_idxes = conv_spaced_list(np.arange(1, FIRE_FRAME_CNT), FIRE_RECONFIRM_CNT, is_end=False)
            else:
                self.fps = 0
                self.frame_list = []
                self.next_frames = deque()
                self.prev_frames = deque()
                self.write_frames = []

                #self.clear()
        except Exception as e:
            print('status', e)
            print(traceback.format_exc())

            pass

        self.frame = None
        self.frame_cnt = 0
        self.dt_update = time.time()

        self.state = 'connect' if is_play else 'disconnect'
        if is_play != self.is_play:
            # print('state>>', self.cctv_id, self.state)
            log.debug('cctv name {} >> {}'.format(self.cctv_name, self.state))
            db['main'].set_cctv_state(self.cctv_id, self.state, self.get_preview_url(), system.get_port())
        self.is_play = is_play

    def on_frame(self, args):
        if self.is_stopped:
            return

        frame, frame_cnt, grab_cnt = args
        # print(f'frame.shape: {frame.shape}')

        if grab_cnt in self.read_idxes:
            self.frame_cnt, self.frame = frame_cnt, frame

            # if IS_WRITE_USE:
            #     if len(self.next_frames) >= FRAME_NEXT_CNT:
            #         self.frame_cnt, self.frame = self.next_frames.popleft()
            #     self.next_frames.append([frame_cnt, frame.copy()])

            # if len(self.frame_list) >= FIRE_FRAME_CNT:
            #     tmp = self.frame_list.pop(0)
            #     del tmp
            # self.frame_list.append(frame.copy())

            # if frame_cnt >= FRAME_PREV_CNT:
            # if self.frame is not None:
            #     if frame_cnt % 20 == 0:
            #         if len(self.prev_frames) > FRAME_PREV_CNT:
            #             self.prev_frame_cnt, self.prev_frame = self.prev_frames.popleft()
            #         self.prev_frames.append([self.results, self.frame.copy()])

            self.dt_update = time.time()

            if time.time() - self.dt_broadcast <= 3:
                self.disp_frame = self.frame.copy()
                self.disp_frame = self.draw_layers(self.disp_frame, self.layer_dic)
                self.disp_frame = self.draw_results(self.disp_frame)
                cv2.putText(self.disp_frame, str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 1)

            # self.tmp_buffer.append(frame.copy())
            # if len(self.tmp_buffer) >= self.fps * int(BEFORE_EVENT_VID_LEN):
            #     self.tmp_buffer.pop(0)
            # if self.recording:
            #     self.evt_vid_buffer.append(frame.copy())
            # if self.start_record:
            #     self.start_record = False
            #     self.recording = True
            #     ymd = datetime.today().strftime('%Y%m%d')
            #     hour = datetime.today().strftime('%H')
            #     directory = os.path.join("record", ymd, hour)
            #     os.makedirs(directory, exist_ok=True)

            #     filename = self.get_filename()
            #     self.jpg_filepath = os.path.join(directory, f"{filename}.jpg")
            #     self.mp4_filepath = os.path.join(directory, f"{filename}.mp4")
            #     cv2.imwrite(self.jpg_filepath, self.frame)

            #     if not self.evt_vid_buffer:
            #         self.evt_vid_buffer = self.tmp_buffer.copy()

            # if self.last_detect_time:
            #     if time.time() - self.last_detect_time > BEFORE_EVENT_VID_LEN:
            #         # self.object_count = 0
            #         if self.recording:
            #             height, width, layers = self.tmp_buffer[0].shape

            #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #             # out = cv2.VideoWriter(self.mp4_filepath, fourcc, self.fps[model], (width, height))
            #             out = cv2.VideoWriter(self.mp4_filepath, fourcc, self.fps, (width, height))
            #             for frame in self.tmp_buffer + self.evt_vid_buffer:
            #                 out.write(frame)
            #             out.release()
            #             self.tmp_buffer = []
            #             self.evt_vid_buffer = []
            #             self.recording = False
            #         self.train = None

            self.tmp_buffer.append(frame.copy())
            if len(self.tmp_buffer) >= self.fps * int(BEFORE_EVENT_VID_LEN):
                self.tmp_buffer.pop(0)
            if self.recording:
                self.recording_frame_cnt += 1
                self.video_writer.add_frame(frame.copy())
                if time.time() > self.detection_time + 5: # or self.recording_frame_cnt >= self.fps * EVENT_VIDEO_LENGTH:
                    self.recording_frame_cnt = 0
                    self.recording = False
                    self.video_writer.stop()
                    self.video_writer = None
                    self.detection_time = None
                    self.object_count = -1
                    
            if self.start_record:
                self.recording = True
                self.start_record = False
                ymd = datetime.today().strftime('%Y%m%d')
                hour = datetime.today().strftime('%H')
                directory = os.path.join("record", ymd, hour)
                os.makedirs(directory, exist_ok=True)

                filename = self.get_filename()
                self.jpg_filepath = os.path.join(directory, f"{filename}.jpg")
                self.mp4_filepath = os.path.join(directory, f"{filename}.mp4")
                cv2.imwrite(self.jpg_filepath, self.frame)

                height, width, _ = frame.shape
                self.video_writer = VideoWriter(self.mp4_filepath, self.fps, (width, height))
                self.video_writer.start()

                for buffered_frame in self.tmp_buffer:
                    self.video_writer.add_frame(buffered_frame)

        if (len(self.detect_idxes[grab_cnt]) > 0
                and detector.is_load):
            detector.buffer.append({
                'worker': self,
                'frame': frame.copy(),
                'frame_cnt': frame_cnt,
                'models': self.detect_idxes[grab_cnt],
                'thresh': {
                    # 'fire':self.fire_confidence,
                    # 'smoke':self.smoke_confidence,
                    'train':self.train_confidence
                }
            })
        del frame

    def draw_layers(self, frame, layer_dic):
        for key, layer in layer_dic.items():
            if len(layer) > 1:
                layer = np.array(layer, dtype=np.int32)
                cv2.polylines(frame, [layer], True, (0, 0, 255), 2)

        return frame

    def draw_results(self, frame):
        # Draw detector lines and object count
        if self.lines:
            for i in range(len(self.lines) - 1):
                start_point = tuple(self.lines[i])
                end_point = tuple(self.lines[i+1])
                cv2.line(frame, start_point, end_point, (0, 255, 255), 2)

            # Calculate midpoint of the line to place the count
            if len(self.lines) >= 2:
                mid_x = (self.lines[0][0] + self.lines[-1][0]) // 2
                mid_y = (self.lines[0][1] + self.lines[-1][1]) // 2

                # Draw object count near the middle of the line
                if hasattr(detector, 'object_count'):
                    count_text = f"{0 if detector.object_count == -1 else detector.object_count}"
                    text_size = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    text_x = mid_x - text_size[0] // 2
                    text_y = mid_y - 10  # Slightly above the line
                    cv2.putText(frame, count_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw bounding boxes and labels
        for result in self.results:
            if result['score'] > TRAIN_SCORE_THRESHOLD and result['hit_frame'] > TRAIN_HIT_FRAME_THRESHOLD:
                label = result['label']
                score = result['score']
                xmin, ymin, xmax, ymax = result['bbox']

                color = (0, 255, 0) if label != 'fire' else (0, 0, 255)
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                cv2.putText(frame, f"{label}: {score:.2f}", (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        return frame

    def get_preview_url(self):
        return 'http://{}:{}/display/{}'.format(SERVER_IP, system.get_port(), self.cctv_id)

    def get_meta(self, uid):
        if uid not in self.meta:
            self.meta[uid] = {
                'is_in_area': False,
                'hit_frame':0,
                'fire_list': [],
                'event_list': [],
                'is_fire': False,
                'fire_cnt': 0,
            }
        return self.meta[uid]

    def get_filename(self):
        date_time = datetime.today().strftime('%Y%m%d%H%M%S%f')+str(random.randint(100, 200))
        return '{}_{}'.format(self.cctv_id, date_time)


class YoloDetector(threading.Thread):
    def __init__(self, module='darknet'):
        threading.Thread.__init__(self)
        self.buffer = deque()
        self.module = module
        self.models = {}
        self.load_list = []
        self.is_load = False
        self.object_count = -1
        self.fps = None

    def run(self):
        while True:
            if self.is_load:
                break

            for load_data in self.load_list:
                name, model, gpus = load_data
                print('on gpus', gpus)
                print('===============================================================')

                self.models[name] = YOLO("model/ultralytics/{}/model.pt".format(model))

            # with open('points.json', 'r', encoding='utf-8') as file:
            #     lines_data = json.load(file)
            #     if str(self.cctv_id) in lines_data.keys():
            #         worker.lines = lines_data[str(self.cctv_id)]

            self.is_load = True
            time.sleep(0.01)

        while True:
            if len(self.buffer) > DETECTION_BUFFER_CNT:
                print(f'clear buffer (len: {len(self.buffer)})')
                self.buffer = deque()

            if len(self.buffer) > 0:
                data = self.buffer.popleft()
                worker = data['worker']
                results = []
                frame = data['frame']

                for model, option in list(DETECTION_OPTIONS.items()):
                    if (model in data['models']
                            and option['type'] == 'main'):
                        if model not in self.models:
                            continue
                        if self.models[model] is None:
                            continue

                        try:
                            if worker.lines is None:
                                height, width, _ = worker.frame.shape
                                worker.lines = [[int(width/2), 0], [int(width/2), height]]
                            else:
                                # poi = time.time() #model inference time
                                output = self.models[model](frame, device="0", verbose=False)
                                # print(f"model inference time: {time.time() - poi}")
                                results = worker.tracker.update(output[0].boxes, frame)

                                analyzer.buffer.append({
                                    'worker': worker,
                                    'frame': data['frame'].copy(),
                                    'results': copy.deepcopy(results),
                                })
                                worker.results = results
                            
                                # for result in results:
                                #     score = result['score']
                                #     hit_frame = result['hit_frame']


                                #     if score > TRAIN_SCORE_THRESHOLD and hit_frame > 1:
                                #         if not worker.recording:
                                #             worker.start_record = True
                                #         worker.last_detect_time = time.time()

                                #         if not result['cross']:
                                #             move_path = result['move_path']
                                #             if is_inout(worker.lines[0], worker.lines[1], move_path[-1], move_path[0]):
                                #                 if self.object_count == -1:
                                #                     worker.train = result['label']
                                #                 self.object_count += 1
                                #                 result['number'] = self.object_count
                                #                 result['cross'] = True

                                #         elif result['cross'] and result['fire']:
                                #             print("FIRE", result['number'])

                        except Exception as e:
                            print('detector>>',  e)
                            pass

            time.sleep(0.001)

    def add_model(self, name, model, gpus=0):
        print('on gpus', gpus)
        print('===============================================================')
        self.models[name] = YOLO("model/ultralytics/{}/model.pt".format(model))


# class YoloDetector2(threading.Thread):
#     def __init__(self, module='darknet'):
#         threading.Thread.__init__(self)
#         self.buffer = deque()
#         self.module = module
#         self.models = {}
#         self.load_list = []
#         self.is_load = False

#     def run(self):
#         while True:
#             if self.is_load:
#                 break

#             for load_data in self.load_list:
#                 name, model, gpus = load_data
#                 print('on gpus', gpus)
#                 print('===============================================================')
#                 # self.models[name] = DarknetWrapper('model/{}/model.cfg'.format(model),
#                 #                         'model/{}/model.weights'.format(model),
#                 #                         'model/{}/model.names'.format(model),
#                 #                         gpus=gpus)
#                 self.models[name] = Yolox(
#                     exp_file='model/yolox/{}/model.py'.format(model),
#                     trt_file='model/yolox/{}/model.pth'.format(model),
#                     size=800
#                 )
#             self.is_load = True
#             time.sleep(0.01)

#         while True:
#             if len(self.buffer) > DETECTION_BUFFER_CNT:
#                 self.buffer = deque()

#             if len(self.buffer) > 0:
#                 data = self.buffer.popleft()
#                 worker = data['worker']
#                 results = []
#                 bind_results = []
#                 obj_height = worker.obj_height

#                 for model, option in list(DETECTION_OPTIONS.items()):
#                     if (model in data['models']
#                             and option['type'] == 'main'):
#                         # model_results = self.models[model].detect(data['frame'], option['thresh'], option['thresh'])
#                         if model not in self.models:
#                             continue
#                         if self.models[model] is None:
#                             continue

#                         new_results = []
#                         try:
#                             ''' yolox '''
#                             model_results = self.models[model].detect(data['frame'], option['thresh'])
#                             for result in model_results:
#                                 label = result[0]
#                                 conf = result[1]
#                                 obj_y = result[2][-1]

#                                 if (conf >= data['thresh'][label]\
#                                     and obj_y  >= obj_height):
#                                     new_results.append(result)
#                         except Exception as e:
#                             print('detector>>',  e)
#                             pass

#                         worker.model_results[model] = new_results
#                     results.extend(worker.model_results[model])
#                 results = worker.tracker.update(results, data['frame'].shape[:2])

#                 # analyzer.buffer.append({
#                 #     'worker': worker,
#                 #     'frame': data['frame'].copy(),
#                 #     'results': copy.deepcopy(results),
#                 #     'bind_results': copy.deepcopy(bind_results),
#                 # })

#                 del results
#                 del data['frame']

#             time.sleep(0.001)

#     def add_model(self, name, model, gpus=0):
#         print('on gpus', gpus)
#         print('===============================================================')
#         self.models[name] = Yolox(
#                                 exp_file='model/yolox/{}/model.py'.format(model),
#                                 trt_file='model/yolox/{}/model.pth'.format(model),
#                                 size=704
#                             )


class EventAnalyzer(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.buffer = deque()

        self.init_ntforg = ntforg.NotificationOriginator()
        self.light_on = '1'

    def run(self):
        while True:
            if len(self.buffer) > 0:
                data = self.buffer.popleft()
                frame = data['frame']
                worker = data['worker']
                results = data['results']               
                for track in results:
                    if track['hit_frame'] < 5:
                        continue
                    if track['score'] < 0.25:
                        continue 
                       
                    # if (time.time() - track['alive']) < 1:
                    #     continue
                    if track['move_dist'] > 60:
                        if not worker.recording: # and worker.detection_time is None
                            worker.start_record = True
                        if not result['cross']:
                            move_path = result['move_path']
                            if is_inout(worker.lines[0], worker.lines[1], move_path[-1], move_path[0]):
                                if self.object_count == -1:
                                    worker.train = result['label']
                                    worker.object_count += 1
                                    track['number'] = self.object_count
                                    track['cross'] = True
                                track['cross'] = True
                        worker.detection_time = time.time()
                    
            time.sleep(0.01)



# class EventAnalyzer(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.buffer = deque()

#         self.init_ntforg = ntforg.NotificationOriginator()
#         self.light_on = '1'

#     def run(self):
#         while True:
#             if len(self.buffer) > 0:
#                 data = self.buffer.popleft()
#                 frame = data['frame']
#                 worker = data['worker']

#                 obj_duration = worker.duration
#                 if not isinstance(worker.duration, int):
#                         obj_duration = 0


#                 for track in data['results']:
#                     uid = track['id']
#                     label = track['label']
#                     xmin, ymin, xmax, ymax = track['bbox']
#                     meta = worker.get_meta(uid)

#                     if 'is_hit' in track:
#                         del track['is_hit']

#                     is_in_area = True
#                     if time.time() - track['maintain_time'] >= obj_duration:
#                         for key, layer in worker.layer_dic.items():
#                             if len(layer) > 0:
#                                 is_in_area = is_polygon_inside(track['bbox_pos'], layer)
#                     else:
#                         is_in_area = False

#                     meta['is_in_area'] = is_in_area
#                     if meta['is_in_area']:
#                         if not meta['is_fire']:
#                             is_fire = self.fire_reconfirm(worker, frame, track)
#                             meta['fire_list'].append(is_fire)

#                             if len(meta['fire_list']) > 4:
#                                 fire_state, _ = Counter(meta['fire_list']).most_common(1)[0]
#                                 if fire_state:
#                                     meta['event_list'].append(fire_state)
#                                 else:
#                                     if len(meta['event_list']) > 0:
#                                         meta['event_list'].pop(0)
#                                 meta['fire_list'] = []
#                                 if len(meta['event_list']) > 2:
#                                     try:
#                                         is_fire, _ = Counter(meta['event_list']).most_common(1)[0]
#                                         meta['event_list'].pop(0)
#                                         if round(time.time() - worker.notify_time, 2) > worker.interval:
#                                             # print(round(time.time() - worker.notify_time, 2))
#                                             worker.notify_time = time.time()
#                                             if is_fire:
#                                                 snmp_info = worker.snmp
#                                                 is_send = self.send_snmp(snmp_info)
#                                                 log.debug('{} SNMP {}'.format(worker.cctv_name, is_send))

#                                                 if worker.http_url != '':
#                                                     requests.get(worker.http_url, timeout=0.1)
#                                                     log.debug('cctv name "{}" >> {} requests'.format(worker.cctv_name, worker.http_url))

#                                         meta['is_fire'] = True
#                                         if meta['is_fire']:
#                                             if not worker.is_detect:
#                                                 worker.write_frames = worker.prev_frames.copy()
#                                                 if time.time() - worker.writer_time >= 30:
#                                                     worker.is_detect = True
#                                                     worker.write_filename = worker.get_filename()
#                                                     worker.writer_time = time.time()
#                                                     print('='*50)

#                                                     roi_frame = data['frame'][ymin:ymax, xmin:xmax].copy()
#                                                     cv2.imwrite(os.path.join('/var/www/html/uploads/', 'thumb', '{}.jpg'.format(worker.write_filename)), data['frame'])
#                                                     # cv2.imwrite(os.path.join('/var/www/html/uploads/', 'detect', '{}.jpg'.format(worker.write_filename)), roi_frame)
#                                                     del roi_frame

#                                         meta['fire_cnt'] = 15
#                                     except Exception as ex:
#                                         log.error('CCTV "{}" >> {} requests fail'.format(worker.cctv_name, worker.http_url))
#                                         meta['event_list'] = []
#                                         pass
#                         else:
#                             if meta['fire_cnt'] > 0:
#                                 meta['fire_cnt'] -= 1
#                             else:
#                                 meta['is_fire'] = False
#                 worker.results = data['results']

#                 try:
#                     del data['frame']
#                 except:
#                     pass
#                 try:
#                     del frame
#                 except:
#                     pass
#             time.sleep(0.001)

#     def fire_reconfirm(self, client, frame=None, result=[], use_gaussian=False, r_size=128, gradient=0.03):
#         '''
#             'client': Class [BomWorker()],
#             'frame': np [현재프레임],
#             'result': list [검출 과],
#             'use_gaussian':bool [가우시안 필터 적용 여부],
#             'r_size':int [이미지 리사이즈 값],
#             'gradient':float [변화도 임계치]
#         '''
#         f1 = None
#         f2 = None
#         frame_list = []
#         dt = time.time()
#         try:
#             frame_list = copy.copy(client.frame_list)#.copy()
#             if (frame is not None
#                     and len(frame_list) >= FIRE_FRAME_CNT):
#                 pre_value = None
#                 dt = time.time()
#                 frame_h, frame_w = frame.shape[:2]

#                 xmin, ymin, xmax, ymax = result['bbox']
#                 xmin = max(xmin, 0)
#                 ymin = max(ymin, 0)
#                 xmax = min(frame_w, xmax)
#                 ymax = min(frame_h, ymax)

#                 f1 = frame[ymin:ymax, xmin:xmax].copy()
#                 f1 = cv2.resize(f1, (r_size, r_size), interpolation=cv2.INTER_NEAREST)
#                 f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)

#                 if use_gaussian:
#                     f1 = cv2.GaussianBlur(f1, (0, 0), 1.0)

#                 ck_list = [] # fire gradient check list
#                 for idx in [0, -1]:
#                     prev_frame = frame_list[int(idx)][ymin:ymax, xmin:xmax].copy()
#                     f2 = cv2.resize(prev_frame, (r_size, r_size), interpolation=cv2.INTER_NEAREST)
#                     f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

#                     if use_gaussian:
#                         f2 = cv2.GaussianBlur(f2, (0, 0), 1.0)

#                     diff_frame = cv2.absdiff(f1, f2)
#                     _, diff_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)

#                     num_of_black_pix = np.sum(diff_frame==0)
#                     if pre_value is None:
#                         pre_value = num_of_black_pix
#                     fire_gradient = round(abs(num_of_black_pix-pre_value)/num_of_black_pix, 2)

#                     if fire_gradient >= gradient:
#                         return True
#                         continue
#         except Exception as e:
#             print('err', e)
#             pass
#         #print(1/(time.time()-dt))
#         if f1 is not None:
#             del f1
#         if f2 is not None:
#             del f2
#         del frame_list
#         return False

#     def send_snmp(self, snmp_info):
#         ip, port, oid = snmp_info['ip'], snmp_info['port'], snmp_info['oid']

#         is_snmp_send = 'send fail'
#         if (ip != '' and port != '' and oid != ''):
#             print(ip, port, oid)
#             try:
#                 errorIndication = self.init_ntforg.sendNotification(
#                     ntforg.CommunityData('public', mpModel=0),
#                     ntforg.UdpTransportTarget((ip, port)),
#                     'trap',
#                     oid,
#                     (oid, rfc1902.OctetString('\n'.join(self.light_on)))
#                 )
#                 is_snmp_send = 'send'
#                 if errorIndication is not None:
#                     log.debug(errorIndication)
#             except Exception as ex:
#                 # log.error('SNMP {}'.format(ex))
#                 # log.debug('CCTV "{}" >> {} requests fail'.format(worker.cctv_name, worker.http_url))
#                 is_snmp_send = ex
#                 pass
#         return is_snmp_send
#         # return is_snmp_success
#         # if is_snmp_success:
#         #     return is_snmo




class BomSystem(threading.Thread):
    def __init__(self, id, gpus=0):
        threading.Thread.__init__(self)
        self.broadcast = None
        self.id = id
        self.gpus = gpus
        self.port_list = []
        self.work_list = []

        self.system_day = datetime.today().strftime('%Y%m%d')
        self.system_hour = datetime.today().strftime('%H')

        self.hit_buffer = deque()
        self.event_buffer = deque()

    def run(self):
        dt_date = time.time()
        dt_frame = time.time()
        dt_event = time.time()

        while True:
            if time.time()-dt_date >= DELAY_DATE_CHECK:
                dt_date = time.time()
                self.refresh_date_change()

            if time.time()-dt_frame >= 1:
                dt_frame = time.time()
                self.check_frame_update()

            if time.time()-dt_event >= DELAY_EVENT_CHECK:
                dt_event = time.time()
                self.check_event()

            time.sleep(1)

    def init_port(self):
        start_port = SERVER_PORT+((self.id+1)*CLIENT_PORT_SPLIT)
        if start_port == SERVER_PORT:
            start_port += 1
        self.port_list = [i for i in range(start_port, start_port+CLIENT_PORT_SPLIT)]

    def get_port(self):
        return random.choice(self.port_list)

    def init_broadcast(self):
        for port in self.port_list:
            if self.broadcast is None:
                self.broadcast = BroadcastServer(app, port)
            else:
                self.broadcast.add_port(port)
        self.broadcast.start()

    def refresh_date_change(self):
        is_day_change = False
        is_hour_change = False

        if self.system_day != datetime.today().strftime('%Y%m%d'):
            self.system_day = datetime.today().strftime('%Y%m%d')
            print('date on change!')
            is_day_change = True
            is_hour_change = True

        for worker in self.work_list:
            if worker.is_play:
                for layer in worker.count_layers:
                    if is_day_change:
                        for key, value in layer['count_day'].items():
                            layer['count_day'][key] = 0
                    if is_hour_change:
                        for key, value in layer['count_hour'].items():
                            layer['count_hour'][key] = 0
                    db['main'].set_layer_counter(worker.cctv_id, layer['layer_id'], layer['count_hour'])

    def check_frame_update(self):
        for i, worker in enumerate(self.work_list):
            if (worker.is_play
                    and worker.frame_cnt >= 60
                    and time.time()-worker.dt_update >= 10):
                worker.dt_update = time.time()
                worker.close()
                worker.connect()
                print('='*50)
                print(worker.cctv_id, 'is dead')

    def check_event(self):
        if len(self.event_buffer) > 0:
            data = self.event_buffer.popleft()

            args = data['args']
            if 'frame' in data:
                tmp_time = datetime.today().strftime('%Y%m%d%H%M%S%f')+str(random.randint(100, 200))
                filename = '{}_{}.jpg'.format(args['cctv_id'], tmp_time)

                xmin, ymin, xmax, ymax = data['bbox']
                cv2.rectangle(data['frame'], (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.polylines(data['frame'], data['draw_poly'], True, data['draw_color'], 2)

                cv2.imwrite('./picture/{}'.format(filename), data['frame'])
                args['picture'] = filename
                db['main'].set_layer_event(args)
                del data['frame']

    def add_worker(self, cctv_id):
        cctv_data = db['main'].get_cctv_data(cctv_id)
        if cctv_data is not None:
            worker = BomWorker(data=cctv_data)
            worker.connect()
            self.work_list.append(worker)

    def delete_worker(self, cctv_id, is_state=False):
        for i, worker in enumerate(self.work_list):
            if worker.cctv_id == cctv_id:
                if is_state:
                    db['main'].set_cctv_state(worker.cctv_id, 'disconnect', worker.get_preview_url(), system.get_port())
                worker.close()
                del self.work_list[i]
                return True
        return False

    def get_bind_worker(self, cctv_id):
        for worker in self.work_list:
            if worker.cctv_id == cctv_id:
                return worker
        return None


@emitter.on('redis_receive')
def on_redis_receive(channel, msg):
    try:
        if channel == 'client':
            data = msg['data']
            if data['client_id'] != system.id:
                return

            if msg['header'] == 'add_cctv':
                system.add_worker(data['cctv_id'])
            elif msg['header'] == 'cctv_modify':
                print('on cctv modify!')
                if system.delete_worker(data['cctv_id']):
                    system.add_worker(data['cctv_id'])
            elif msg['header'] == 'cctv_work':
                system.delete_worker(data['cctv_id'], is_state=True)
            elif msg['header'] == 'cctv_delete':
                worker = system.get_bind_worker(data['cctv_id'])
                if worker is not None:
                    worker.close()
                    del worker
            elif msg['header'] == 'cctv_restart':
                worker = system.get_bind_worker(data['cctv_id'])
                if worker is not None:
                    worker.close()
                    worker.connect()
                    print('{} restart cctv!'.format(data['cctv_id']))
            elif msg['header'] == 'layer_modify':
                worker = system.get_bind_worker(data['cctv_id'])
                if worker is not None:
                    worker.set_layer_list()
                    print('{} modify layer!'.format(data['cctv_id']))
            elif msg['header'] == 'guest_delete':
                guest_delete(data['guest_name'], is_db=False)
            elif msg['header'] == 'change_variable':
                init_site_option()
            else:
                print(channel, msg)

    except Exception as e:
        log.error('client:', e)


app = Flask(__name__)
@app.route('/thumbnail/<int:cctv_id>')
def broadcast_thumbnail(cctv_id=-1):
    return Response(get_broadcast_thumbnail(int(cctv_id)), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_broadcast_thumbnail(cctv_id):
    worker = system.get_bind_worker(cctv_id)
    if worker is None:
        frame = BROADCAST_NOSIGNAL_IMG.copy()
    else:
        frame = BROADCAST_NOSIGNAL_IMG.copy() if worker.frame is None else worker.frame.copy()

    frame = imutils.resize(frame, width=BROADCAST_THUMB_WIDTH)

    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    buf = jpeg.tobytes()
    return (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + buf + b'\r\n\r\n')

@app.route('/display/<int:cctv_id>')
def broadcast_display(cctv_id=-1):
    args = request.args.to_dict()
    disp_type = args['type'] if 'type' in args else 'thumb'

    if disp_type == 'thumb':
        return Response(get_broadcast_thumbnail(int(cctv_id)), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(get_broadcast_display(system, int(cctv_id), disp_type), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_argument():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('id', type=int, help='client id')
    parser.add_argument('gpus', type=int, help='gpu id')
    return parser.parse_args()



log = Logger(disp_list=[])

args = get_argument()

log.info('{} id on {} gpus'.format(args.id, args.gpus))

print(args.id, '=================')
print(args.gpus, '=========================')
system = BomSystem(args.id, args.gpus)
#system = BomSystem(0 ,0)
system.init_port()
system.init_broadcast()
system.start()


detector = YoloDetector()
for model, option in list(DETECTION_OPTIONS.items()):
    detector.load_list.append([model, option['model'], args.gpus])
    detector.add_model(name=model, model=option['model'], gpus=args.gpus)
detector.start()

analyzer = EventAnalyzer()
analyzer.start()

#display = BomDisplay(is_prev=False, is_write=False)
#display.start()

db = {}
db['main'] = db_connect(DB_CONF['local'], reconn_sec=DB_RECONN_DELAY)

# system.tool_list = db['main'].get_tool_list()

redis = {}
redis['local'] = redis_connect(REDIS_CONF['local'], subscribe='subscribe_client', reconn_sec=REDIS_RECONN_DELAY)
redis['local'].bind_client(system.id)

print('client>>', system.id, system.gpus)
dt = time.time()


while True:
    # print(system.tool_list )
    # print('send ping', len(system.work_list), '->', len(detector.buffer), '\t', len(analyzer.buffer))
    print('send ping', len(system.work_list), '->', len(detector.buffer))
    redis['local'].send_ping(system.id)
    time.sleep(CLIENT_PING_DELAY)
