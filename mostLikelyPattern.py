import os
import cv2
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
DEBUG = 1
DEBUG2 = 1

base_path = Path(__file__).parent

def get_files_from_dir(dir, format = ['jpg', 'jpeg', 'png', 'tif', 'tiff']):
    lst_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.split('.')[-1].lower() in format:
                lst_files.append(os.path.join(root, file))
    return lst_files

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def merge_boxes(boxes):
    remove = []
    for i in range(len(boxes) - 1):
        for j in range(i+1, len(boxes)):
            if boxes[j][0] > boxes[i][0] + boxes[i][2]:
                break
            if boxes[i][0] <= boxes[j][0] and boxes[i][0] + boxes[i][2] >= boxes[j][0] + boxes[j][2]:
                remove.append(boxes[j])
    pick = [cnt for cnt in boxes if cnt not in remove]
    return pick

# TEMPLATE MATCHING
def im2bin(im):
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_im, 127, 255, 0)
    vertical = np.copy(thresh)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    vertical = cv2.dilate(vertical, verticalStructure)
    return vertical

def trim_image(im, bbox):
    cnts = np.array([np.array([x, y, x+w, y+h]) for x, y, w, h in bbox])
    xmin, ymin, _, _ = np.min(cnts, 0)
    _, _, wmax, hmax = np.max(cnts, 0)
    im = im[ymin:hmax, xmin:wmax]
    # update bbox
    cnts[:, 0] -= xmin
    cnts[:, 2] -= xmin
    cnts[:, 1] -= ymin
    cnts[:, 3] -= ymin
    return im, cnts 

def remove_noise(im, bbox):
    # Create mask
    mask = np.zeros((im.shape[0], im.shape[1]), dtype = np.uint8)
    for box in bbox:
        cv2.rectangle(mask,(box[0], box[1]),(box[2], box[3]),255,-1)

    # Put white
    mask = cv2.bitwise_not(mask)
    im[(mask==255)] = (255,255,255)

    return im

def candle_resize(im, candle_w=15):    
    thresh = im2bin(im)
    # Get contours
    contours_, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # filter small boundingbox
    # contours_ = [box for box in contours_ if cv2.contourArea(box) > 10]
    _, contours = sort_contours(contours_)
    contours = [cnt for cnt in contours if cnt[2] < im.shape[1]//2]
    contours = merge_boxes(contours)

    # crop & remove background
    im, cnts = trim_image(im, contours)
    im = remove_noise(im, cnts)

    # resize
    theWidth = len(contours) * candle_w
    new_im = image_resize(im, width=theWidth) # resize keep ratio
    # new_im = cv2.resize(new_im, (new_im.shape[1], int(new_im.shape[0])), interpolation=cv2.INTER_LINEAR) # stretch im's height *0.7 | *1.4

    if DEBUG:
        print('num contours:', len(contours), '- candle_w:', candle_w, '- new_shape:', new_im.shape)
        im_debug = im.copy()
        print()
        for x, y, xm, ym in cnts:
            cv2.rectangle(im_debug,(x,y),(xm,ym),(255,0,255),1)
        
        cv2.imshow('bin', thresh)
        cv2.imshow('bb', im_debug)
        cv2.waitKey(0)
    return new_im

def _findThePattern(year_data, template):
    bigIm = year_data['img']
    gray = cv2.cvtColor(bigIm, cv2.COLOR_BGR2GRAY)      
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    score = max_val / (bigIm.shape[0]  * bigIm.shape[1])
    height, width = template.shape[:2]
    bottom_right= (max_loc[0] + width, max_loc[1] + height)

    if DEBUG2:
        im_debug = bigIm.copy()
        im_debug = cv2.putText(im_debug, str(year_data['date']) + ' __ ' + str(score), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
        3, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.rectangle(im_debug, max_loc, bottom_right, (255, 255, 0),2)
        cv2.imshow('im_debug', cv2.resize(im_debug, (im_debug.shape[1]//3, im_debug.shape[0]//3), interpolation = cv2.INTER_AREA))
        cv2.waitKey(0)

    return score, (max_loc[0], max_loc[1], width, height)

def findThePattern(pattern, top_k=5, padding=10):
    # Template preprocessing
    template = cv2.copyMakeBorder(pattern, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = (255, 255, 255))
    template = candle_resize(template, candle_w=15)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template2 = cv2.resize(template, (template.shape[1], int(template.shape[0]*0.7)), interpolation=cv2.INTER_LINEAR)

    # Template matching
    results = [[key, _findThePattern(value, template)] for key, value in yearly_data.items()]
    results2 = [[key, _findThePattern(value, template2)] for key, value in yearly_data.items()]
    results.extend(results2)
    
    # Result Post-processing
    results = sorted(results, key=lambda x: x[1][0], reverse=True)
    results = results[:top_k]
    final_result = []
    for key, rs in results:
        period = find_date(yearly_data[key] , rs[1], padding=padding)
        # result ['year data name', 'score', 'loc', period]
        final_result.append([key, rs[0], rs[1], period])
    return final_result


# DATE PROCESSING ===
def date_by_adding_business_days(from_date, add_days):
    business_days_to_add = add_days
    current_date = from_date
    while business_days_to_add > 0:
        current_date += timedelta(days=1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date

def find_date(year_data, loc, padding=10):
    w_can = year_data['img'].shape[1] / year_data['num']

    num_start = loc[0] // w_can
    num_end = (loc[0]+loc[2]) // w_can

    start_date = date_by_adding_business_days(year_data['date'], num_start)
    end_date = date_by_adding_business_days(year_data['date'], num_end)

    # Padding 
    start_date -= timedelta(days=padding)
    end_date += timedelta(days=padding)
    return start_date, end_date

def load_yearly_data(dir):
    yearly_data = {}
    for file in get_files_from_dir(dir):
        # load image
        im = cv2.imread(file)
        # load txt
        txt_dir = file.replace(file.split('.')[-1], 'txt')
        if os.path.exists(txt_dir):
            with open(txt_dir, 'r') as f:
                lines = f.readlines()
                num_candles = int(lines[0].strip())
                start_date = lines[1].strip()
                start_date = datetime.strptime(start_date, '%d-%m-%Y').date()
            yearly_data[file.split('/')[-1]] = {'num':num_candles, 'img':im, 'date':start_date}
        else:
            file = file.split('/')[-1].split('_')
            if len(file) == 4:
                num_candles = int(file[-1].split('.')[0])
                start_date = datetime.strptime(file[1].replace('.', '-'), '%Y-%m-%d').date()
                yearly_data['_'.join(file[:-1])] = {'num':num_candles, 'img':im, 'date':start_date}
            elif len(file) == 3:
                num_candles = int(file[-1].split('.')[0])
                start_date = datetime.strptime(file[1].replace('.', '-'), '%Y-%m-%d').date()
                yearly_data['_'.join(file[:-1])] = {'num':num_candles, 'img':im, 'date':start_date}

    return yearly_data

yearly_data_path = os.path.join(base_path, 'Year_data/year_data/XAUUSD3/')
yearly_data = load_yearly_data(yearly_data_path)
print('========== Yearly data loaded ==========', len(yearly_data))


if __name__ == "__main__":
    
    DEBUG = 1
    DEBUG2 = 0

    for file in get_files_from_dir(os.path.join(base_path, 'test/XAUUSD')):
        # if 'XAUUSD_1118_0419_NG' not in file: continue
        # XAUUSD_1119_0420
                
        print(file)
        # Read input template
        template = cv2.imread(file)
        print('size:', template.shape, template.shape[0] * template.shape[1])

        # FIND THE PATTERN FROM DATABASE
        t0 = time.time()
        rs = findThePattern(template, top_k=3, padding=0)
        print('total time:', time.time() - t0)
    
        # Visualize
        for i, (key, score, loc, period) in enumerate(rs):
            print(f'{period[0]}-{period[1]} : {score}')
            if DEBUG:
                im_debug = yearly_data[key]['img'].copy()
                text = f'{period[0]} to {period[1]} : {score}'
                im_debug = cv2.putText(im_debug, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                x, y, w, h = loc
                cv2.rectangle(im_debug, (x, y), (x+w, y+h), (0, 255, 0),4)
                cv2.imshow('im_debug', cv2.resize(im_debug, (im_debug.shape[1]//3, im_debug.shape[0]//3), interpolation = cv2.INTER_AREA))
                cv2.waitKey(0)
                
        print('--')
        cv2.destroyAllWindows()     
