import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import numpy as np
from get_license_plate import get_license_plate


class UI(object):
    window = tk.Tk()
    window.resizable(width=True, height=True)
    filename = ''
    frame = tk.Frame(master=window, width=550, height=250)
    frame.pack()
    frame_a = tk.Frame()
    label_license = tk.Label(master=frame_a, text="")
    label_license.pack()

    def openfn(self):
        filename = filedialog.askopenfilename(title='open')
        return filename

    def open_img(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.label_license.config(text="")
        self.filename = self.openfn()
        img = Image.open(self.filename)
        img.thumbnail((500, 500), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self.frame, image=img)
        panel.image = img
        panel.pack()

    def display_img(self, img):
        for widget in self.frame.winfo_children():
            widget.destroy()
        img.thumbnail((500, 500), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(self.frame, image=img)
        panel.image = img
        panel.pack()

    def process_img(self):
        print(self.filename)
        res = get_license_plate(self.filename)

        self.label_license.config(text=res["plate_options"])
        self.display_img(Image.fromarray(np.uint8(res["image"])).convert('RGB'))

    def __init__(self):

        label_license = tk.Label(master=self.frame_a, text="")
        label_license.pack()
        label_a = tk.Button(master=self.frame_a, command=self.open_img, text="Load image")
        label_a.pack(side=tk.LEFT)

        frame_b = tk.Frame()
        label_b = tk.Button(master=frame_b, command=self.process_img, text="Recognize plate")
        label_b.pack(side=tk.LEFT)

        self.frame_a.pack()
        frame_b.pack()

        self.window.mainloop()

if __name__ == "__main__":
    ui = UI()


