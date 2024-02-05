from tkinter import filedialog, PhotoImage
from PIL import Image, ImageTk, ImageDraw
import customtkinter
from FixCTkCanvas import FixCTkCanvas
import webbrowser
from segmentation.segment import init_segment_model, segment_image
from MAT import generate_images, init_gan_model
from CTkMessagebox import CTkMessagebox
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Easy Image Inpaint.')

parser.add_argument('--resolution', type=int, required=True, help='Resolution as an integer')
parser.add_argument('--sam_name', type=str, required=True, help='SAM name as a string')
parser.add_argument('--sam_pretrain', type=str, required=True, help='SAM pretrain path as a string')
parser.add_argument('--mat_pretrain', type=str, required=True, help='MAT pretrain path as a string')

args = parser.parse_args()

class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, file_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Edit Image")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        self.image = Image.open(file_path)

        # Check if the image height is greater than the screen height
        if self.image.height > screen_height:
            # Calculate new dimensions while maintaining the aspect ratio
            aspect_ratio = self.image.width / self.image.height
            new_height = screen_height
            new_width = int(aspect_ratio * new_height)

            # Resize the image
            self.image = self.image.resize((new_width, new_height))

        # Create the PhotoImage object with the resized image
        self.tk_image = ImageTk.PhotoImage(self.image)
        
        self.geometry(f"{self.image.width}x{self.image.height}")
        
        self.button_frame = customtkinter.CTkFrame(self)
        self.button_frame.pack(fill='x', side='top')
        
        self.canvas = MaskCanvas(self, self.image)
        self.canvas.pack(fill='both', expand=True)



class MaskCanvas(FixCTkCanvas):
    def __init__(self, master, image):
        super().__init__(master)

        self.master.resizable(False, False)

        self.create_mask_mode = 'rec'

        self.image = image
        self.bg_image = ImageTk.PhotoImage(self.image)
        self.ontop_image = None
        self.mask = None

        self.create_image(0, 0, anchor='nw', image=self.bg_image)

        self.update_bindings()
        
        self.points = []
        self.rectangle = None

        self.brush_size = 4

        self.rec_draw_button = customtkinter.CTkButton(self.master.button_frame, text="Move and Replace", command=self.rec_draw_mode, fg_color='#8d591f', hover=False)
        self.rec_draw_button.pack(side='left', fill='x', expand=True, padx=10)
        
        self.free_draw_button = customtkinter.CTkButton(self.master.button_frame, text="Remove an object", command=self.free_draw_mode, fg_color='#1f538d', hover=True)
        self.free_draw_button.pack(side='left', fill='x', expand=True, padx=10)


    def rec_draw_mode(self):
        self.create_mask_mode = 'rec'
        self.update_bindings()
        self.rec_draw_button.configure(fg_color="#8d591f")
        self.rec_draw_button.configure(hover=False)
        self.free_draw_button.configure(fg_color="#1f538d")
        self.free_draw_button.configure(hover=True)

    def free_draw_mode(self):
        self.create_mask_mode = 'freedraw'
        self.update_bindings()
        self.rec_draw_button.configure(fg_color="#1f538d")
        self.rec_draw_button.configure(hover=True)
        self.free_draw_button.configure(fg_color="#8d591f")
        self.free_draw_button.configure(hover=False)
        self.focus_set()

    def update_bindings(self):
        # Unbind all events
        self.unbind("<Button-1>")
        self.unbind("<Motion>")
        self.unbind("<B1-Motion>")
        self.unbind("<MouseWheel>")
        self.unbind("<Button-4>")
        self.unbind("<Button-5>")
        self.unbind("<Return>")
    
        # Bind specific events based on the create_mask_mode
        if self.create_mask_mode == 'rec':
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)
            
            self.bind("<Button-1>", self.on_click)
            self.bind("<Motion>", self.on_mouse_move)
        elif self.create_mask_mode == 'freedraw':            
            self.bind("<B1-Motion>", self.free_draw_mask)
            self.bind("<MouseWheel>", self.brush_size_adjust)  # For Windows
            self.bind("<Button-4>", self.brush_size_adjust)  # For Unix
            self.bind("<Button-5>", self.brush_size_adjust)  # For Unix
            self.bind("<Return>", self.confirm_free_drawn_mask) # enter to confirm
        elif self.create_mask_mode == 'segment':
            self.bind('<Motion>', self.on_mouse_move_for_mask)
            self.bind("<Button-1>", self.send_to_GAN)

    
    def free_draw_mask(self, event):
        oval_id = self.create_oval((event.x - self.brush_size / 2, event.y - self.brush_size / 2, event.x + self.brush_size / 2, event.y + self.brush_size / 2), fill='black', tags='oval')
        
    def brush_size_adjust(self, event):
        # For Windows, 'MouseWheel' event returns positive delta when scrolling up
        if event.num == 'MouseWheel':
            if event.delta > 0:
                self.brush_size += 4
            else:
                # Ensure brush_size doesn't go below 0
                if self.brush_size > 0:
                    self.brush_size -= 4
        # For Unix, 'Button-4' and 'Button-5' events are for scrolling up and down
        elif event.num == 4:
            self.brush_size += 4
        elif event.num == 5:
            # Ensure brush_size doesn't go below 0
            if self.brush_size > 0:
                self.brush_size -= 4

    def confirm_free_drawn_mask(self, event):
        msg = CTkMessagebox(title='Confirmation', message='Do you want to use this mask ?', icon='question', option_1='Cancle', option_2='No', option_3='Yes')
        if msg.get() == 'No':
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)
            self.points = []
            self.create_mask_mode = 'freedraw'
            self.update_bindings()
        elif msg.get() == 'Yes':
            new_mask = Image.new('L', (self.image.width, self.image.height), 'white')
            draw = ImageDraw.Draw(new_mask)
            for oval in self.find_withtag('oval'):
                x1, y1, x2, y2 = self.coords(oval)
                draw.ellipse([(x1, y1), (x2, y2)], fill='black')

            new_mask = np.array(new_mask)
            kernel = np.ones((7,7),np.uint8)
            eroded_image = cv2.erode(new_mask, kernel, iterations = 10)
            new_mask = Image.fromarray(eroded_image)
            
            gen_image = generate_images(self.image, new_mask, mat_model, resolution=args.resolution)
            self.bg_image = ImageTk.PhotoImage(gen_image)
            
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)

            self.create_mask_mode = 'gan'
            self.update_bindings()

            self.rec_draw_button.destroy()
            self.free_draw_button.destroy()
            
            self.save_button = customtkinter.CTkButton(self.master.button_frame, text="Save Image", command=self.on_save_result)
            self.save_button.pack(side='left', fill='x', expand=True, padx=10)
            
    def on_mouse_move_for_mask(self, event):
        self.create_image(0, 0, anchor='nw', image=self.bg_image)
        self.create_image(event.x, event.y, anchor='nw', image=self.ontop_image)

    def send_to_GAN(self, event):
        msg = CTkMessagebox(title='Confirmation', message='Do you want to remove segmentation or not ?', icon='question', option_1='Cancle', option_2='No', option_3='Yes')
        if msg.get() == 'No':
            ontop_image = ImageTk.getimage(self.ontop_image)
            self.image.paste(ontop_image, (event.x, event.y), mask=ontop_image)
            gen_image = generate_images(self.image, self.mask, mat_model, resolution=args.resolution)
            self.bg_image = ImageTk.PhotoImage(gen_image)
            
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)

            self.create_mask_mode = 'gan'
            self.update_bindings()

            self.rec_draw_button.destroy()
            self.free_draw_button.destroy()

            self.save_button = customtkinter.CTkButton(self.master.button_frame, text="Save Image", command=self.on_save_result)
            self.save_button.pack(side='left', fill='x', expand=True, padx=10)
            
        elif msg.get() == 'Yes':
            
            gen_image = generate_images(self.image, self.mask, mat_model, resolution=args.resolution)
            self.bg_image = ImageTk.PhotoImage(gen_image)
            
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)

            self.create_mask_mode = 'gan'
            self.update_bindings()

            self.rec_draw_button.destroy()
            self.free_draw_button.destroy()

            self.save_button = customtkinter.CTkButton(self.master.button_frame, text="Save Image", command=self.on_save_result)
            self.save_button.pack(side='left', fill='x', expand=True, padx=10)
            
            
    def on_save_result(self):
        image_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG Files", "*.png")])
        if image_path:
            bg_image = ImageTk.getimage(self.bg_image)
            bg_image.save(image_path)
                
    def on_click(self, event):
        self.points.append((event.x, event.y))
        if len(self.points) == 2:
            self.draw_rectangle(*self.points[0], *self.points[1], width=3, outline='red')
            self.confirm_mask()
            
    def on_mouse_move(self, event):
        if self.points:
            if self.rectangle is not None:
                self.delete(self.rectangle)
            x1, y1 = self.points[0]
            x2, y2 = event.x, event.y
            self.rectangle = self.create_rectangle(x1, y1, x2, y2, outline='red', width=3)
    
    def on_drag(self, event):
        x, y = event.x, event.y
        self.coords(self.ontop_image, x, y)
        
    def draw_rectangle(self, x1, y1, x2, y2, **kwargs):
        rectangle_id = self.create_rectangle(x1, y1, x2, y2, **kwargs)
        return rectangle_id

    def confirm_mask(self):
        msg = CTkMessagebox(title='Confirmation', message='Do you want to use this mask ?', icon='question', option_1='Cancle', option_2='Confirm')
        if msg.get() == 'Confirm':
            x1, y1, x2, y2 = *self.points[0], *self.points[1]
            image_without_segment, segmented_image, mask = segment_image(segment_predictor, self.image, x1, y1, x2, y2)
            
            self.bg_image = ImageTk.PhotoImage(image_without_segment)
            self.ontop_image = ImageTk.PhotoImage(segmented_image)
            self.mask = mask
            
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)
            
            self.points = []
            self.create_mask_mode = 'segment'
            self.update_bindings()

        else:
            self.points = []
            self.delete("all")
            self.create_image(0, 0, anchor='nw', image=self.bg_image)

    
class TextFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        photo = customtkinter.CTkImage(dark_image=Image.open("images/github.png"), size=(30, 30))
        
        button = customtkinter.CTkButton(self, text="Have Fun 😊", image=photo, compound='right', command=self.open_link)
        button.place(relx=0.5, rely=0.5, anchor='center')

    def open_link(self):
        webbrowser.open_new("https://github.com/achira-kati/easy-image-inpaint")
         

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("300x200")
        customtkinter.set_default_color_theme("dark-blue")
        self.title("Easy Image Inpaint")
        
        self.grid_rowconfigure(0, weight=1)  # configure grid system
        self.grid_columnconfigure(0, weight=1)

        self.my_frame = TextFrame(master=self)
        self.my_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.button = customtkinter.CTkButton(self, text='Upload Image', command=self.load_image)
        self.button.grid(row=1, column=0, padx=10, pady=10, sticky="we")

        self.toplevel_window = None
        
    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select An Image", filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
        if file_path:
            if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
                self.toplevel_window = ToplevelWindow(file_path)  # create window if its None or destroyed
            else:
                self.toplevel_window.focus()  # if window exists focus it
        

if __name__ == "__main__":
    segment_predictor = init_segment_model(args.sam_name, args.sam_pretrain)
    mat_model = init_gan_model(resolution=args.resolution, network_pkl=args.mat_pretrain)
    
    app = App()
    app.mainloop()