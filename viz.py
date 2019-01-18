from PIL import Image
import sys
from PIL import Image
import sys

def blend_two_images():
    for i in range(3):
        i = str(i)    #.zfill(6)
        print(i)
        img_add = '/home/zgx010/Desktop/blendimgtest/'+ i +'.jpg'
        img1 = Image.open(img_add)
        img1 = img1.convert('RGBA')
        label_add = '/home/zgx010/Desktop/blendimgtest/'+ i +'_1.png'
        img2 = Image.open(label_add)
        img2 = img2.convert('RGBA')

        img = Image.blend(img1, img2, 0.3)
        #img.show()
        img.save('/home/zgx010/Desktop/blendimgtest/'+ i +'_2.png')
        sys.stdout.write('\r>> Converting image %d/%d' % (int(i), 119))
        sys.stdout.flush()
    print '\r'
    return

if __name__ == '__main__':
    blend_two_images()
