from PIL import Image

frames = []

for frame_number in range(1,3881):
    frame = Image.open(f'./results/video/{frame_number}.jpg')
    frames.append(frame)

frames[0].save(
    'video_pred.gif',
    save_all=True,
    append_images=frames[1:],  
    optimize=True,
    duration=60,
    loop=0
)