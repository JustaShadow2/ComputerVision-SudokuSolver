# DataFlair Sudoku solver

import cv2
import numpy as np
import imutils
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
from solver import *

classes = np.arange(0, 10) # 0-9 digits

model = load_model('model-OCR.h5') # load the pre trained model
print(model.summary()) # print the model summary
input_size = 48 # input size of the model


def get_perspective(img, location, height = 900, width = 900):
    """Takes an image and location os interested region.
        And return the only the selected region with a perspective transformation"""
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def get_InvPerspective(img, masked_num, location, height = 900, width = 900):
    """Takes original image as input"""
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result

def find_board(img):
    """Takes an image as input and finds a sudoku board inside of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20) # apply bilateral filter to smooth the image
    edged = cv2.Canny(bfilter, 30, 180) # apply canny edge detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours
    contours  = imutils.grab_contours(keypoints) # grab contours

    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3) # draw contours on the original image
    cv2.imshow("Contour", newimg)  # show contours


    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15] # sort contours from big to small and get only the first 15 contours
    location = None
    
    # Finds rectangular contour
    for contour in contours: 
        approx = cv2.approxPolyDP(contour, 15, True) # approximate the contour
        if len(approx) == 4: # if the contour has 4 corners then it's a rectangle
            location = approx # save the location of the rectangle
            break   # break the loop if the rectangle is found
    result = get_perspective(img, location) # get the perspective of the rectangle
    return result, location # return the rectangle and its location


# split the board into 81 individual images
def split_boxes(board):
    """Takes a sudoku board and split it into 81 cells. 
        each cell contains an element of that board either given or an empty cell."""
    rows = np.vsplit(board,9) # split the board into 9 rows
    boxes = [] # list to store the cells
    for r in rows: # for each row
        cols = np.hsplit(r,9) # split the row into 9 columns
        for box in cols: # for each column
            box = cv2.resize(box, (input_size, input_size))/255.0 # resize the image to 48x48 and normalize it
            cv2.imshow("Splitted block", box) # show the splitted block 
            cv2.waitKey(1000) # wait for 1000ms
            boxes.append(box) # append the block to the list
    cv2.destroyAllWindows() 
    return boxes # return the list of cells

'''for solver'''
# def displayNumbers(img, numbers, color=(0, 255, 0)):
#     """Displays 81 numbers in an image or mask at the same position of each cell of the board"""
#     W = int(img.shape[1]/9) # width of each cell
#     H = int(img.shape[0]/9) # height of each cell
#     for i in range (9): # for each row
#         for j in range (9): # for each column
#             if numbers[(j*9)+i] !=0: # if the number is not 0 then it's a given number
#                 cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA) # display the number in the cell
#     return img

# Read image, sudoku1 and sudoku4 are the only two that work so far, maybe sudoku2
img = cv2.imread('sudoku4.jpg')


# extract board from input image
board, location = find_board(img)


gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
print(gray.shape)
rois = split_boxes(gray)
rois = np.array(rois).reshape(-1, input_size, input_size, 1)

# get prediction
prediction = model.predict(rois)
print(prediction)

predicted_numbers = []
# get classes from prediction
for i in prediction: 
    index = (np.argmax(i)) # returns the index of the maximum number of the array
    predicted_number = classes[index]
    predicted_numbers.append(predicted_number)

print(predicted_numbers)

# reshape the list 
board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)



# solve the board
# try:
#     solved_board_nums = get_board(board_num)

#     # create a binary array of the predicted numbers. 0 means unsolved numbers of sudoku and 1 means given number.
#     binArr = np.where(np.array(predicted_numbers)>0, 0, 1)
#     # print(binArr)
#     # get only solved numbers for the solved board
#     flat_solved_board_nums = solved_board_nums.flatten()*binArr
#     # create a mask
#     mask = np.zeros_like(board)
#     # displays solved numbers in the mask in the same position where board numbers are empty
#     solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
#     # cv2.imshow("Solved Mask", solved_board_mask)
#     inv = get_InvPerspective(img, solved_board_mask, location)
#     # cv2.imshow("Inverse Perspective", inv)
#     combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
#     cv2.imshow("Final result", combined)
#     # cv2.waitKey(0)
    

# except:
#     print("Solution doesn't exist. Model misread digits.")

# cv2.imshow("Input image", img)
cv2.imshow("Board", board)
cv2.waitKey(0)
cv2.destroyAllWindows()