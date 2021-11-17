import cv2
import numpy as np

from PIL import Image



# def viewImage(image, name_of_window):
#     cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
#     cv2.imshow(name_of_window, image)
#     cv2.setMouseCallback(name_of_window, onMouse)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    

i = 0
elem = 1801
# k = 1800/100
while i < elem:
    im1 = Image.open('./prepare/gen/bigarrow/12.png')
    im2 = Image.open('./prepare/gen/bigarrow/4.png')
    im3 = Image.open('./prepare/gen/bigarrow/3.png')
#     imnoice = Image.open('./prepare/gen/bigarrow/noice8.png')
#     if i == 0*k:
#         imstem = Image.open('./prepare/gen/stem/0.png')
#     elif i <= 3*k:
#         imstem = Image.open('./prepare/gen/stem/3.png')
#     elif i <= 5*k:
#         imstem = Image.open('./prepare/gen/stem/5.png')
#     elif i <= 8*k:
#         imstem = Image.open('./prepare/gen/stem/8.png')
#     elif i <= 10*k:
#         imstem = Image.open('./prepare/gen/stem/10.png')
#     elif i <= 14*k:
#         imstem = Image.open('./prepare/gen/stem/14.png')
#     elif i <= 20*k:
#         imstem = Image.open('./prepare/gen/stem/20.png')
#     elif i <= 24*k:
#         imstem = Image.open('./prepare/gen/stem/24.png')
#     elif i <= 26*k:
#         imstem = Image.open('./prepare/gen/stem/26.png')
#     elif i <= 30*k:
#         imstem = Image.open('./prepare/gen/stem/30.png')
# # 
#     elif i <= 35*k:
#         imstem = Image.open('./prepare/gen/stem/35.png')
#     elif i <= 40*k:
#         imstem = Image.open('./prepare/gen/stem/40.png')
#     elif i <= 48*k:
#         imstem = Image.open('./prepare/gen/stem/48.png')
#     elif i <= 50*k:
#         imstem = Image.open('./prepare/gen/stem/50.png')
#     elif i <= 55*k:
#         imstem = Image.open('./prepare/gen/stem/55.png')
#     elif i <= 58*k:
#         imstem = Image.open('./prepare/gen/stem/58.png')
#     elif i <= 60*k:
#         imstem = Image.open('./prepare/gen/stem/60.png')
#     elif i <= 70*k:
#         imstem = Image.open('./prepare/gen/stem/70.png')
#     elif i <= 75*k:
#         imstem = Image.open('./prepare/gen/stem/75.png')
#     elif i <= 77*k:
#         imstem = Image.open('./prepare/gen/stem/77.png')
#     elif i <= 80*k:
#         imstem = Image.open('./prepare/gen/stem/80.png')
#     elif i <= 84*k:
#         imstem = Image.open('./prepare/gen/stem/84.png')
#     elif i <= 90*k:
#         imstem = Image.open('./prepare/gen/stem/90.png')
#     elif i <= 95*k:
#         imstem = Image.open('./prepare/gen/stem/95.png')
#     elif i <= 102*k:
#         imstem = Image.open('./prepare/gen/stem/100.png')
    im2 = im2.rotate((0.02)*i, center=(1613, 1114))
    im1.paste(im2,(0,0),im2)
    im3 = im3.rotate(-0.2*i, center=(1290,958))
    im1.paste(im3,(0,0),im3)
    # im1.paste(imnoice,(0,0),imnoice)
    # im1.paste(imstem,(0,0),imstem)
    im1.save('./img2/test'+str(i)+'.png')
    i+=1

# im2 = im2.rotate(0.1*100, center=(1613, 1114))
# im1.paste(im2,(0,0),im2)
# im3 = im3.rotate(-3.6*47, center=(1290,958))
# im1.paste(im3,(0,0),im3)

# im1.save('test1.png')


# def onMouse(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#        # draw circle here (etc...)
#        print('x = %d, y = %d'%(x, y))


# img1 = cv2.imread('test1.png')
# img2 = cv2.imread('./prepare/gen/bigarrow/4.png', cv2.IMREAD_COLOR)



# img = cv2.imread('./prepare/gen/bigarrow/4.png', cv2.IMREAD_UNCHANGED)
# (h, w, d) = img2.shape
# # center = (w // 2, h // 2)
# # center = (1615, 1112)
# # M = cv2.getRotationMatrix2D(center, 13, 1.0)
# # rotated = cv2.warpAffine(img2, M, (w, h))

# dst = cv2.addWeighted(img1, 1, img2, 1, 0.0)

# viewImage(img1,'Picture')



