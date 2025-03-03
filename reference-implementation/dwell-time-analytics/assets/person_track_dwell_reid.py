import sys
import gi
gi.require_version('GstVideo', '1.0')
gi.require_version('Gst', '1.0')
gi.require_version('GObject', '2.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GLib, GstVideo, GObject, GstApp
from gstgva import VideoFrame, util
import cv2 as cv
import numpy as np
from datetime import datetime
import copy
import math
import json
from munkres import Munkres

# init GStreamer
Gst.init(sys.argv)
# Global Variables
obj_id = []
dtime = {}
a = 0
map_img = cv.imread("/home/dlstreamer/Floor_Plan_v5.jpg")
hg_mat = None
cam_id = None
identities = []
reid_threshold = 0.7
matcher = Munkres()

def calc_person_map_location(ref_point):
    global hg_mat
    proj_person_loc = (0,0,0)
    ref_point_homo = (ref_point[0], ref_point[1], 1)
    proj_person_loc = np.dot(hg_mat, ref_point_homo)
    proj_person_loc = (proj_person_loc[0]/proj_person_loc[2], proj_person_loc[1]/proj_person_loc[2],1)
    return proj_person_loc

def cam_to_floor_map(bbox_coord):
    global map_img      
    global a
    fp_img = copy.deepcopy(map_img)
    for c in bbox_coord:
        person_map_loc = calc_person_map_location(c)
        x,y = int(person_map_loc[0]),int(person_map_loc[1])         
        cv.circle(fp_img,(x,y),6,(255,0,0),thickness=2)        
    if a%60 == 0:
        cv.imshow("Floor Plan",fp_img)
    return True 

def compute_reid_distance(test_embedding, reference_embedding):
    xx = np.dot(test_embedding, test_embedding)
    yy = np.dot(reference_embedding, reference_embedding)
    xy = np.dot(test_embedding, reference_embedding)
    norm = math.sqrt(xx * yy) + 1e-6
    return np.float32(1.0) - xy / norm

def get_ids_by_reid(frame):
    global identities
    global matcher
    global reid_threshold
    detected_tensors = []
    detection_ids = []
    match_list = []
    messages = list(frame.messages())
    if len(messages) > 0:
        json_msg = json.loads(messages[0])
        frame_ts = int(json_msg["timestamp"]) / 1000000000

    regions = [x for x in frame.regions()]
    for i, roi in enumerate(regions):        
        if roi.label_id() == 1:            
            for j, tensor in enumerate(roi.tensors()):
                if tensor.layer_name() == "reid_embedding":
                    detected_tensors.append(tensor.data())
                    detection_ids.append(i)                    
    
    if len(detected_tensors) == 0:
        return
    if len(identities) == 0:
        for i in range(len(detected_tensors)):
            identities.append({"embedding": copy.deepcopy(detected_tensors[i]), "timestamp": frame_ts})
        return
    distances = np.empty([len(detected_tensors), len(identities)], dtype=np.float32)
    for i in range(len(detected_tensors)):
        for j in range(len(identities)):
            distances[i][j] = compute_reid_distance(detected_tensors[i], identities[j]["embedding"])
    matched_indexes = matcher.compute(distances.tolist())
    matched_detections = set()
    # Update the timestamp for the matched person
    for match in matched_indexes:
        if distances[match[0]][match[1]] <= reid_threshold:
            identities[match[1]]["timestamp"] = frame_ts
            matched_detections.add(match[0])
            match_list.append(match)
    
    # Add the newly detected person into identities
    for i in range(len(detected_tensors)):
        if i not in matched_detections:
            #print("New item added to Identities at index ",len(identities))
            identities.append({"embedding": copy.deepcopy(detected_tensors[i]), "timestamp": frame_ts})
            match_list.append((i,len(identities)))
    # Update the tensor_id of reid_embedding layer to the matched index in identities
    for k, roi in enumerate(frame.regions()):
        for tensor in roi.tensors():
            if tensor.layer_name() == "reid_embedding" and k == match_list[k][0]:
                tensor["tensor_id"] = match_list[k][1]


def process_frame(frame: VideoFrame):
    global obj_id
    global a
    global dtime
    global cam_id
    global identities
    if a < 1:
        print("Frame Rate: ", frame.video_info().fps_n)
    box_coord = []    
    xy = None    
    get_ids_by_reid(frame)   

    with frame.data() as mat:
        a += 1        
        for roi in frame.regions():
            person_id = None
            for tensor in roi.tensors():
                if tensor.layer_name() == "reid_embedding":
                    person_id = tensor["tensor_id"]
            
            rect = roi.rect()
            # Computing the bottom-centre coordinates of the bounding box  
            xy = (int(rect.x + rect.w/2),int(rect.y + rect.h))
            box_coord.append(xy)            
            det_obj_id = person_id      
            if not (det_obj_id in obj_id):
                obj_id.append(det_obj_id)                
                dtime[det_obj_id] = [datetime.now(),0]
            else:
                current_ts = datetime.now()
                person_duration = (current_ts - dtime[det_obj_id][0]).total_seconds()
                dtime[det_obj_id][0] = current_ts
                dtime[det_obj_id][1] += person_duration
            
            for tensor in roi.tensors():
                if tensor.label_id() == 1:                                        
                    cv.putText(mat, "ID: {}, Time: {}s".format(det_obj_id,int(dtime[det_obj_id][1])), (rect.x + 50, rect.y-10), cv.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)                    
                    cv.circle(mat, xy, 8,(255,0,0),thickness=2)
        # Uncomment the following two lines to display the movement of persons on the floor-plan image
        '''                  
        if box_coord:
            cam_to_floor_map(box_coord)
        '''

def pad_probe_callback(pad, info):
    with util.GST_PAD_PROBE_INFO_BUFFER(info) as buffer:
        caps = pad.get_current_caps()        
        frame = VideoFrame(buffer, caps=caps)
        process_frame(frame)
    return Gst.PadProbeReturn.OK

def glib_mainloop():
    mainloop = GLib.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        pass

def bus_call(bus, message, pipeline):
    t = message.type
    global dtime    
    if t == Gst.MessageType.EOS:
        print("pipeline ended")
        #dtime = sorted(dtime.items())
        for k in dtime:
            if dtime[k][1] >= 1:
                print("Person ID: {}, Dwell Time: {} sec".format(k,int(dtime[k][1])))
        print("Frame Count: ",a)                
        pipeline.set_state(Gst.State.NULL)
        cv.destroyAllWindows()
        sys.exit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print("Error:\n{}\nAdditional debug info:\n{}\n".format(err, debug))
        pipeline.set_state(Gst.State.NULL)
        sys.exit()
    else:
        pass
    return True


def set_callbacks(pipeline):
    gvawatermark = pipeline.get_by_name("gvawatermark")
    pad = gvawatermark.get_static_pad("src")
    pad.add_probe(Gst.PadProbeType.BUFFER, pad_probe_callback)
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, pipeline)


if __name__ == '__main__':

    Gst.init(sys.argv)
    inp_video = sys.argv[1]
    frame_interval = sys.argv[2]
    cam_id = 1 # Input arg3
    cam_map_corners = {1: {"cam":[(249,94),(422,94),(0,400),(610,400)],"map":[(211,86), (446,86), (211,396), (446,396)]}}    
    # Calculate the Homography matrix between camera and map corners
    hg_mat = cv.getPerspectiveTransform(np.float32(cam_map_corners[cam_id]["cam"]),np.float32(cam_map_corners[cam_id]["map"]))
    
    pipeline_str = "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=BGRx ! queue ! " \
            "gvadetect model=/home/dlstreamer/models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml inference-interval={} threshold=0.8 device=CPU ! queue ! " \
            "gvaclassify model=/home/dlstreamer/models/intel/person-reidentification-retail-0277/FP16/person-reidentification-retail-0277.xml reclassify-interval={} device=GPU ! queue ! " \
            "gvametaconvert ! " \
            "gvawatermark name=gvawatermark ! videoconvert ! fpsdisplaysink video-sink=xvimagesink sync=True".format(inp_video,frame_interval,frame_interval)
    
    print(pipeline_str)
    # Uncomment the following line to display the floor-plan image
    #cv.imshow("Floor Plan",map_img)
    
    pipeline = Gst.parse_launch(pipeline_str)

    set_callbacks(pipeline)

    pipeline.set_state(Gst.State.PLAYING)

    glib_mainloop()

    print("Exiting")

