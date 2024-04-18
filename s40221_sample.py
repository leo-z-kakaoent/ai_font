import sys
from PIL import Image
from font_diffuser.args import SampleArgs
from font_diffuser.sample import load_fontdiffuer_pipeline, sampling

class ImageConcat:
    def __init__(self, n_h, n_v, resolution=96):
        self.n_h = n_h
        self.n_v = n_v
        self.r = resolution
        self.img = Image.new("RGB", (self.n_h*self.r, self.n_v*self.r))
        self.cursor = [0,0]
        
    def move_cursor_h(self):
        self.cursor[0] += self.r
        
    def move_cursor_v(self):
        self.cursor[1] += self.r
        
    def reset_cursor_h(self):
        self.cursor[0] = 0
        
    def append_img(self, im):
        self.img.paste(im, tuple(self.cursor))
        
    def save_img(self, path):
        self.img.save(path)

def main(style, style_i):
    args = SampleArgs()
    r = 96
    content_is = range(0,11172,1100)
    bigimg = ImageConcat(n_h=(len(content_is)+1), n_v=3+4, resolution=r)

    bigimg.append_img(Image.open("data/f40219/content.png").convert('RGB'))
    for i in content_is:
        bigimg.move_cursor_h()
        bigimg.append_img(Image.open("data/r40202/pngs/gulim__%s.png"%str(i)).convert('RGB'))
    bigimg.reset_cursor_h()
    bigimg.move_cursor_v()

    bigimg.append_img(Image.open("data/f40219/style.png").convert('RGB'))
    for i in content_is:
        bigimg.move_cursor_h()
        bigimg.append_img(Image.open("data/r40202/pngs/%s__%s.png"%(str(style), str(style_i))).convert('RGB'))
    bigimg.reset_cursor_h()
    bigimg.move_cursor_v()

    bigimg.append_img(Image.open("data/f40219/target.png").convert('RGB'))
    for i in content_is:
        bigimg.move_cursor_h()
        try:
            bigimg.append_img(Image.open("data/r40202/pngs/%s__%s.png"%(str(style), str(i))).convert('RGB'))
        except:
            bigimg.append_img(Image.open("data/f40219/noimg.png").convert('RGB'))
    bigimg.reset_cursor_h()
    bigimg.move_cursor_v()

    for mi in [0,1,2,3]:
        pipe = load_fontdiffuer_pipeline(args=args,model_i=mi*10000)
        bigimg.append_img(Image.open("data/f40219/phase1_ep%s.png"%str(mi)).convert('RGB'))
        for ci in content_is:
            content_image = Image.open("data/r40202/pngs/gulim__%s.png"%str(ci)).convert('RGB')
            style_image = Image.open("data/r40202/pngs/%s__%s.png"%(str(style), str(style_i))).convert('RGB')
            out_image = sampling(
                args=args, 
                pipe=pipe, 
                content_image=content_image,
                style_image=style_image,
            )
            bigimg.move_cursor_h()
            bigimg.append_img(out_image)
        bigimg.reset_cursor_h()
        bigimg.move_cursor_v()
    bigimg.save_img("data/f40221/%s__%s.png"%(style,style_i))

if __name__=="__main__":
    style = sys.argv[1]
    style_i = sys.argv[2]
    print(style)
    print(style_i)
    main(style, style_i)
    print()
    print("done")
