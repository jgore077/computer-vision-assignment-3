import cv2
from flan import Flanned_Matcher

def keypoint_detect_orb(image,orb):
    """
    Detects keypoints for orb 
    """
    # compute the descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints,descriptors

def keypoint_detect_sift(image,sift):
    """
    Detects keypoints for
    """
    return sift.detectAndCompute(image,None)

def iterate_image_pairs(name:str,image_dict:dict,size:int,detector_type,detector,matcher=None,matcher_type=None):
    for image_path in image_dict:
        img_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_2 = cv2.imread(image_dict[image_path],cv2.IMREAD_GRAYSCALE)

        match detector_type:
            case "ORB":
                keypoints_1,descriptors_1=keypoint_detect_orb(img_1,detector)
                keypoints_2,descriptors_2=keypoint_detect_orb(img_2,detector)
            case "SIFT":
                keypoints_1,descriptors_1=keypoint_detect_sift(img_1,detector)
                keypoints_2,descriptors_2=keypoint_detect_sift(img_2,detector)
                
                
    
        # draw only keypoints location,not size and orientation
        keypoints_img_1 = cv2.resize(cv2.drawKeypoints(img_1, keypoints_1, None, color=(0,255,0), flags=0),(size,size))
        keypoints_img_2 = cv2.resize(cv2.drawKeypoints(img_2, keypoints_2, None, color=(0,255,0), flags=0),(size,size))
        
        # Match descriptors.
        draw_params=None
        matches=None
        # For using BfMatcher
        if not matcher_type:
            matches = matcher.match(descriptors_1,descriptors_2)
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first 10 matches.
            img3 = cv2.drawMatches(keypoints_img_1,keypoints_1,keypoints_img_2,keypoints_2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchesThickness=1)
            cv2.imshow(name,img3)
            cv2.waitKey(0)
        
        # For using Flann matcher
        elif matcher_type=="FLANN":
            draw_params,matches =Flanned_Matcher(descriptors_1,descriptors_2) 
            img3 = cv2.drawMatchesKnn(keypoints_img_1,keypoints_1,keypoints_img_2,keypoints_2,matches,None,**draw_params)
            cv2.imshow(name,img3)
            cv2.waitKey(0)
        # Sort them in the order of their distance.
        
        
    