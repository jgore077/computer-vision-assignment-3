import cv2

#Function returns the draw_params and matches using a ratio test
def Flanned_Matcher(des1,des2,nndr=.8):


	# FLANN parameters.
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50) 

	# FLANN based matcher with implementation of k nearest neighbour.
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# selecting only good matches.
	matchesMask = [[0,0] for i in range(len(matches))]

	# ratio test.
	for i,(m,n) in enumerate(matches):
		if( m.distance < nndr*n.distance):
			matchesMask[i]=[1,0]
	draw_params = dict(matchColor = (0,255,0),
					singlePointColor = (255,0,0),
					matchesMask = matchesMask,flags = 0)
	
	return draw_params,matches
