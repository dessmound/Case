import os
from PIL import Image

path = os.listdir(str(os.getcwd())+'/img3/input')


number = 1

for el in path:
    ind = int(str.split(el,'.')[0])
    j = 1
    while j < 16:
        img = Image.open(str(os.getcwd())+'/img3/input/'+el)
        imnoice = Image.open('./prepare/gen/noice/noice'+str(j)+'.png')
        if ind == 0:
            imstem = Image.open('./prepare/gen/stem/0.png')
        elif ind <= 3:
            imstem = Image.open('./prepare/gen/stem/3.png')
        elif ind <= 5:
            imstem = Image.open('./prepare/gen/stem/5.png')
        elif ind <= 8:
            imstem = Image.open('./prepare/gen/stem/8.png')
        elif ind <= 10:
            imstem = Image.open('./prepare/gen/stem/10.png')
        elif ind <= 14:
            imstem = Image.open('./prepare/gen/stem/14.png')
        elif ind <= 20:
            imstem = Image.open('./prepare/gen/stem/20.png')
        elif ind <= 24:
            imstem = Image.open('./prepare/gen/stem/24.png')
        elif ind <= 26:
            imstem = Image.open('./prepare/gen/stem/26.png')
        elif ind <= 30:
            imstem = Image.open('./prepare/gen/stem/30.png')
        elif ind <= 35:
            imstem = Image.open('./prepare/gen/stem/35.png')
        elif ind <= 40:
            imstem = Image.open('./prepare/gen/stem/40.png')
        elif ind <= 48:
            imstem = Image.open('./prepare/gen/stem/48.png')
        elif ind <= 50:
            imstem = Image.open('./prepare/gen/stem/50.png')
        elif ind <= 55:
            imstem = Image.open('./prepare/gen/stem/55.png')
        elif ind <= 58:
            imstem = Image.open('./prepare/gen/stem/58.png')
        elif ind <= 60:
            imstem = Image.open('./prepare/gen/stem/60.png')
        elif ind <= 70:
            imstem = Image.open('./prepare/gen/stem/70.png')
        elif ind <= 75:
            imstem = Image.open('./prepare/gen/stem/75.png')
        elif ind <= 77:
            imstem = Image.open('./prepare/gen/stem/77.png')
        elif ind <= 80:
            imstem = Image.open('./prepare/gen/stem/80.png')
        elif ind <= 84:
            imstem = Image.open('./prepare/gen/stem/84.png')
        elif ind <= 90:
            imstem = Image.open('./prepare/gen/stem/90.png')
        elif ind <= 95:
            imstem = Image.open('./prepare/gen/stem/95.png')
        elif ind <= 100:
            imstem = Image.open('./prepare/gen/stem/100.png')

        img.paste(imnoice,(0,0),imnoice)
        img.paste(imstem,(0,0),imstem)
        img.save(str(os.getcwd())+'/img3/output/'+str(number)+'_'+str(ind)+'.png')
        j+=1
        number+=1
    j = 0    
