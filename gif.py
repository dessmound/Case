from PIL import Image

frames = []

for frame_number in range(1,184):
    frame = Image.open(f'./results/testwithoutbg/{frame_number}.jpg')
    frames.append(frame)

frames[0].save(
    'test_without_bg.gif',
    save_all=True,
    append_images=frames[1:],  
    optimize=True,
    duration=120,
    loop=0
)