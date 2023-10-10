from tkinter import *
from tkinter import filedialog
from algorithms import *
from PIL import Image,ImageTk
import threading
from queue import Queue
import time
import numpy as np
process_thread = threading.Thread()
msg_queue = Queue()
root = Tk()
root.clrs = []
options_pixelisation = [
    "Nearest",
    "Cubic",
    "Paper"
]

options_quantisation = [
    "None",
    "Median cut",
    "Max coverage",
    "Fast Octree"
]
cur_option_pixelisation = "Nearest"
cur_option_quantisation = "None"
control_pnl = Frame(root)
control_pnl.grid(row=0, column=1, sticky=N, pady=50, padx=50)
img = Image.open("imgs/in.jpg").resize((400,400), Image.NEAREST)
main_img = img.copy()
canv = Canvas(root, width=900, height=500, bg='white')
canv.grid(row=0, column=0)
root.sprpxl_img = None
def displ_results():
    
    Pixeliser.doing = True
    Pixeliser.process(cur_option_pixelisation)
    while Pixeliser.doing or not Pixeliser.msg_queue.empty():
        ret = Pixeliser.msg_queue.get(block=True)
        canv.delete(root.rimg)
        root.rimg = ImageTk.PhotoImage( ret.resize((400,400), Image.NEAREST))
        canv.create_image(440,20, anchor=NW, image=root.rimg)
def on_eps_slide(inp):
    inp = float(inp)
    Pixeliser.eps_c = inp

def on_d_slide(inp):
    inp = float(inp)
    Pixeliser.eps_d = inp

def on_slide(inp):
    inp = int(inp)
    Pixeliser.sz = (inp,inp)
    if cur_option_pixelisation == "Paper":
        return
    displ_results()
def draw_palette(inp):
    inp = int(inp)-1
    if inp >= len(Pixeliser.palette_history):
        inp = -1
    for c in root.clrs:
        canv.delete(c)
    root.clrs = []
    # print(len(Pixeliser.palette_history[inp]))
    for i, c in enumerate(Pixeliser.palette_history[inp]):
        r = canv.create_rectangle(10+30*i, 430,30+30*i,450,fill ="#%02x%02x%02x" % (int(c[0]), int(c[1]), int(c[2])))
        root.clrs.append(r)
    if root.sprpxl_img != None:
        canv.delete(root.sprpxl_img)
    tmp = Image.fromarray(Pixeliser.sprpxl_history[inp//2].reshape(400,400,4).astype(np.uint8), 'RGBA')
    # tmp = Image.fromarray(np.ones((400,400,4), dtype = np.uint8 )*255, 'RGBA')
    root.sprpxl_img = ImageTk.PhotoImage(tmp)
    canv.create_image(20,20, anchor=NW, image=root.sprpxl_img)
def file_browser():
    global img
    filename = filedialog.askopenfilename(initialdir = ".",
                                          title = "Select a File")
    img = Image.open(filename).resize((400,400), Image.NEAREST)
    Pixeliser.glob_img = img
    Pixeliser.quantise()
    canv.delete("all")
    root.limg = ImageTk.PhotoImage(img, Image.NEAREST)
    canv.create_image(20,20, anchor=NW, image=root.limg)
    displ_results()
    
def on_click(inp):
    global cur_option_pixelisation
    cur_option_pixelisation = inp
    print(cur_option_pixelisation)
    displ_results()

def on_click_quant(inp):
    Pixeliser.method = inp
    Pixeliser.quantise()
    displ_results()

def on_slide_quant(inp):
    inp = int(inp)
    Pixeliser.quantisation = inp
    Pixeliser.quantise()
    if cur_option_pixelisation == "Paper":
        return
    
    displ_results()

def on_minimal_change_slider(inp):
    inp = float(inp)
    Pixeliser.minimal_change = inp
def on_sigma_slider(inp):
    inp = float(inp)
    Pixeliser.sigma = inp
clicked_pixelisation = StringVar()
  
clicked_pixelisation.set( cur_option_pixelisation )

clicked_quantisation = StringVar()
  
clicked_quantisation.set( cur_option_quantisation )

drop = OptionMenu( control_pnl , clicked_pixelisation, *options_pixelisation, command= on_click)
drop.grid(row = 1, column = 0, sticky=N)

drop = OptionMenu( control_pnl , clicked_quantisation, *options_quantisation, command= on_click_quant)
drop.grid(row = 3, column = 0, sticky=N)

scl = Scale(control_pnl, from_=8, to=400,orient=HORIZONTAL, command = on_slide, resolution=8, length = 200)
scl.grid(row = 2, column = 0, sticky=N)

scl_quant = Scale(control_pnl, from_=2, to=256,orient=HORIZONTAL, command = on_slide_quant, resolution=2, length = 200)
scl_quant.grid(row = 4, column = 0, sticky=N)
scl_quant.set(8)
btn_fls = Button(control_pnl, text = "Load Image" ,command = file_browser)
btn_fls.grid(row=0, column=0, pady=20)
btn_displ = Button(control_pnl, text = "Display" ,command = displ_results)
btn_displ.grid(row=6, column=0, pady=20)
lbl_eps_var = StringVar()
lbl_eps = Label( control_pnl, textvariable=lbl_eps_var )
lbl_eps_var.set("Epsilon change:")
lbl_eps.grid(row=7, column=0)
scl_eps = Scale(control_pnl, from_=0.0001, to=0.1,orient=HORIZONTAL, command = on_eps_slide, resolution=0.000005, length = 200)
scl_eps.grid(row = 8, column = 0, sticky=N)
scl_eps.set(0.0025)

lbl_d_var = StringVar()
lbl_d = Label( control_pnl, textvariable=lbl_d_var )
lbl_d_var.set("Minimum distance:")
lbl_d.grid(row=9, column=0)
scl_d = Scale(control_pnl, from_=0.0005, to=1,orient=HORIZONTAL, command = on_d_slide, resolution=0.005, length = 200)
scl_d.grid(row = 10, column = 0, sticky=N)
scl_d.set(0.005)

lbl_d_var = StringVar()
lbl_d = Label( control_pnl, textvariable=lbl_d_var )
lbl_d_var.set("Minimum change:")
lbl_d.grid(row=11, column=0)
scl_d = Scale(control_pnl, from_=2, to=8,orient=HORIZONTAL, command = on_d_slide, resolution=0.0005, length = 200)
scl_d.grid(row = 12, column = 0, sticky=N)
scl_d.set(2.5)

lbl_d_var = StringVar()
lbl_d = Label( control_pnl, textvariable=lbl_d_var )
lbl_d_var.set("Bilateral sigma:")
lbl_d.grid(row=13, column=0)
scl_d = Scale(control_pnl, from_=0.01, to=2,orient=HORIZONTAL, command = on_sigma_slider, resolution=0.01, length = 200)
scl_d.grid(row = 14, column = 0, sticky=N)
scl_d.set(0.5)



lbl_p_var = StringVar()
lbl_p = Label( control_pnl, textvariable=lbl_p_var )
lbl_p_var.set("Iter:")
lbl_p.grid(row=19, column=0)
scl_p = Scale(control_pnl, from_=1, to=2000,orient=HORIZONTAL, command = draw_palette, resolution=2, length = 400)
scl_p.grid(row = 20, column = 0, sticky=N)
scl_p.set(1000)

img1 = ImageTk.PhotoImage(img)
canv.create_image(20,20, anchor=NW, image=img1)
Pixeliser.glob_img = img
Pixeliser.quantise()
resized_image = img.resize((8,8), Image.NEAREST)
resized_image = resized_image.resize((400,400), Image.NEAREST)
root.rimg = ImageTk.PhotoImage(resized_image)
canv.create_image(440,20, anchor=NW, image=root.rimg)


mainloop()