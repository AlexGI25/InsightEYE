import tkinter as tk
from tkinter import Label, Button, filedialog, Toplevel, Entry
import cv2
import pyttsx3
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import speech_recognition as sr
import torch
import time
import gc

# initializing text-to-speech
engine = pyttsx3.init()


engine.setProperty('rate', 150)

# verify if we can use GPU instead of CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load trained model
model_path = 'bestV5.pt'
# Actualizează calea către modelul tău
model = YOLO(model_path).to(device)


current_button = None
last_announcements = []
cap = None
target_object = "any"
running = False
button_order = []

# function for selecting voice or text input
def select_input_method(callback):
    def on_up_arrow():
        dialog.destroy()
        ask_object(callback)

    def on_down_arrow():
        dialog.destroy()
        ask_object_voice(callback)

    dialog = Toplevel(root)
    dialog.title("Select Input Method")

    instruction_text = "Press up arrow for text input or down arrow for voice input."
    engine.say(instruction_text)
    engine.runAndWait()

    Label(dialog, text=instruction_text).pack(pady=10)
    dialog.bind("<Up>", lambda event: on_up_arrow())
    dialog.bind("<Down>", lambda event: on_down_arrow())
    dialog.focus_set()

def listen_for_exit_command():
    recognizer = sr.Recognizer()
    while running:
        with sr.Microphone() as source:
            print("Listening for 'exit' command...")
            try:
                audio = recognizer.listen(source, timeout=None)  # Listen indefinitely
                response = recognizer.recognize_google(audio).lower()
                if "exit" in response:
                    stop_running()
                    break
            except sr.UnknownValueError:
                continue  # Continue listening if the speech was not understood
            except sr.RequestError:
                break  # Exit if there is an error with the speech recognition service



# function which requests the user to specify the object he wants to detect
# through text
def ask_object(callback):
    def on_keypress(event):
        char = event.char
        if char == '\x08':  # Backspace character
            engine.say("Backspace")
        elif char.isprintable():
            engine.say(char)
        engine.runAndWait()

    def on_enter():
        global current_button
        if current_button != "Enter":
            engine.say("Enter selected, press again to confirm.")
            engine.runAndWait()
            current_button = "Enter"
        else:
            global target_object
            target_object = entry.get().lower()
            dialog.destroy()
            current_button = None
            if callback == load_video_start:
                engine.say("Uploading video...")
            else:
                engine.say("Live detection in progress")
            engine.runAndWait()
            callback()

    def on_cancel():
        global current_button
        if current_button != "Q":
            engine.say("Q selected, press again to confirm.")
            engine.runAndWait()
            current_button = "Q"
        else:
            target_object = "any"
            dialog.destroy()
            current_button = None
            show_buttons()

    dialog = Toplevel(root)
    dialog.title("Enter Object")

    instruction_text = (
        "Enter the object you want to detect. For example: bowl, cup, bottle, knife, or book. "
        "Or type any to detect all objects. If you are sure with your option, please press enter. "
        "If you want to go to the main menu now or during detection, please press Q."
    )
    engine.say(instruction_text)
    engine.runAndWait()

    Label(dialog, text=instruction_text).pack(pady=10)
    entry = Entry(dialog)
    entry.pack(pady=10)
    entry.focus_set()

    entry.bind("<KeyPress>", on_keypress)  # Binding the keypress event to the entry widget

    entry.bind("<Return>", lambda event: on_enter())  # Binding for Enter key
    entry.bind("<KeyPress-q>", lambda event: on_cancel())  # Binding for Q key

# function which requests the user to specify the object he wants to detect
# through voice
def ask_object_voice(callback):
    def recognize_object():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            engine.say("Please name the object you want to detect. Say any to detect all objects. Anytime during detection, if you want to go back to the main menu, please say exit.")
            engine.runAndWait()
            print("Listening for object...")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio).lower()
            engine.say(f"You said {text}. Say confirm to proceed, again to repeat, or exit to return to the main menu.")
            engine.runAndWait()
            print(f"Recognized: {text}")

            def confirm_or_again():
                with sr.Microphone() as source:
                    print("Listening for confirmation...")
                    audio = recognizer.listen(source)
                try:
                    response = recognizer.recognize_google(audio).lower()
                    if response == "confirm":
                        global target_object
                        target_object = text
                        dialog.destroy()
                        if callback == load_video_start:
                            engine.say("Uploading video...")
                        else:
                            engine.say("Live detection in progress")
                        engine.runAndWait()
                        callback()
                    elif response == "again":
                        recognize_object()
                    elif response == "exit":
                        dialog.destroy()
                        show_buttons()
                    else:
                        engine.say("Invalid response. Say confirm, again, or exit.")
                        engine.runAndWait()
                        confirm_or_again()
                except sr.UnknownValueError:
                    engine.say("Sorry, I did not understand that. Please say confirm, again, or exit.")
                    engine.runAndWait()
                    confirm_or_again()

            confirm_or_again()
        except sr.UnknownValueError:
            engine.say("Sorry, I did not understand that. Please try again.")
            engine.runAndWait()
            recognize_object()

    dialog = Toplevel(root)
    dialog.title("Voice Input")
    threading.Thread(target=recognize_object).start()



# function which detects the object and returns a vocal message
def detect_and_announce(frame):
    results = model(frame)
    announcements = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf > 0.5:  # minimum threshold of 0.5
                cls = int(box.cls.item())
                object_name = model.names[cls].lower()


                if target_object != "any" and target_object != object_name:
                    continue

                announcements.append(f"{object_name} detected")

                # drawing bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{object_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    global last_announcements
    last_announcements = announcements

    for announcement in announcements:
        engine.say(announcement)
        print(announcement)
    engine.runAndWait()

    return frame

# function for repeating the vocal message
def repeat_announcement(event):
    global last_announcements
    for announcement in last_announcements:
        engine.say(announcement)
        print(announcement)
    engine.runAndWait()


def stop_running(event=None):
    global running
    running = False
    if cap:
        cap.release()
    engine.say("Returning to the main menu.")
    engine.runAndWait()
    video_label.config(image='')
    show_buttons()
    # Eliberăm memoria cache GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def hide_buttons():
    open_image_button.grid_remove()
    live_button.grid_remove()
    load_video_button.grid_remove()
    close_button.grid_remove()


def show_buttons():
    open_image_button.grid()
    live_button.grid()
    load_video_button.grid()
    close_button.grid()


def open_image():
    hide_buttons()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = cv2.imread(file_path)

        # resizing the image
        img_resized = cv2.resize(img, (480, 520))


        img_resized = detect_and_announce(img_resized)


        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)


        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)


        canvas.coords(video_label_window, 512, 100)


        engine.say("Press R to hear the prediction again.")
        engine.runAndWait()

# live detection function
def live_detection():
    select_input_method(live_detection_start)

def live_detection_start():
    hide_buttons()
    global cap, running, prev
    running = True
    prev = time.time()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
    frame_rate = 10.0

    def update_frame():
        global prev
        if not running:
            cap.release()
            return
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if ret and time_elapsed > 1.0 / frame_rate:
            prev = time.time()
            frame = detect_and_announce_with_position(frame, target_object)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            canvas.coords(video_label_window, 512, 200)

        video_label.after(10, update_frame)

    threading.Thread(target=update_frame).start()
    threading.Thread(target=listen_for_exit_command).start()



def load_video():
    select_input_method(load_video_start)

def load_video_start():
    hide_buttons()
    global prev
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not file_path:
        show_buttons()
        return

    global cap, running
    cap = cv2.VideoCapture(file_path)
    running = True
    prev = time.time()
    frame_rate = 10.0  # adjusting frame rate to prevent lag in detection

    def update_frame():
        global prev
        if not running:
            cap.release()
            return
        time_elapsed = time.time() - prev
        ret, frame = cap.read()
        if ret and time_elapsed > 1.0 / frame_rate:
            prev = time.time()
            frame = cv2.resize(frame, (416, 416))
            frame = detect_and_announce_with_filter(frame)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)

            canvas.coords(video_label_window, 512, 200)

        if cap.isOpened():
            video_label.after(10, update_frame)

    threading.Thread(target=update_frame).start()
    threading.Thread(target=listen_for_exit_command).start()


# function for detection of the objects in load video option
def detect_and_announce_with_filter(frame):
    results = model(frame)
    announcements = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf > 0.5:
                cls = int(box.cls.item())
                object_name = model.names[cls].lower()


                if target_object != "any" and target_object != object_name:
                    continue

                announcements.append(f"{object_name} detected")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{object_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    global last_announcements
    last_announcements = announcements

    for announcement in announcements:
        engine.say(announcement)
        print(announcement)
    engine.runAndWait()

    return frame


def detect_and_announce_with_position(frame, target_object):
    results = model(frame)
    announcements = []
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    central_zone_left = int(frame_width * 0.3)
    central_zone_right = int(frame_width * 0.7)
    threshold_near = 30000  # threshold to consider the object near the camera

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = box.conf.item()
            if conf > 0.5:
                cls = int(box.cls.item())
                object_name = model.names[cls].lower()

                # verifying if the object is the one user requires
                if target_object != "any" and target_object != object_name:
                    continue

                announcements.append(f"{object_name} detected")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{object_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                object_center_x = (x1 + x2) // 2
                object_area = (x2 - x1) * (y2 - y1)

                if object_center_x < central_zone_left:
                    engine.say(f"{object_name} identified, please move the camera to the left")
                elif object_center_x > central_zone_right:
                    engine.say(f"{object_name} identified, please move the camera to the right")
                elif object_area > threshold_near:
                    engine.say(f"{object_name} object is right here")
                else:
                    engine.say("Move the camera closer")

    global last_announcements
    last_announcements = announcements

    for announcement in announcements:
        engine.say(announcement)
        print(announcement)
    engine.runAndWait()

    return frame


def create_button_functionality(command, description):
    def wrapper():
        global current_button
        if current_button != description:
            engine.say(f"{description} selected, please press again for confirmation.")
            engine.runAndWait()
            current_button = description
        else:
            command()
            current_button = None

    return wrapper

# function whichs creates a gradient on the background
def create_gradient(canvas, width, height, color1, color2):
    gradient = tk.PhotoImage(width=width, height=height)
    for i in range(height):
        r = int(color1[1:3], 16) + (int(color2[1:3], 16) - int(color1[1:3], 16)) * i // height
        g = int(color1[3:5], 16) + (int(color2[3:5], 16) - int(color1[3:5], 16)) * i // height
        b = int(color1[5:7], 16) + (int(color2[5:7], 16) - int(color1[5:7], 16)) * i // height
        color = f'#{r:02x}{g:02x}{b:02x}'
        gradient.put(color, to=(0, i, width, i + 1))
    canvas.create_image((width / 2, height / 2), image=gradient, state="normal")
    canvas.image = gradient


root = tk.Tk()
root.title("Object Detection with YOLOv5")

# setting the user interface dimensions
root.geometry("1024x900")


canvas = tk.Canvas(root, width=1024, height=900)
canvas.pack(fill="both", expand=True)


create_gradient(canvas, 1024, 900, "#87CEFA", "#1E90FF")


button_frame = tk.Frame(canvas, bg="#87CEFA")
button_frame.pack(pady=10)

# plotting the title in the user interface
title_label = Label(canvas, text="InsightEYE", font=("Helvetica", 24, "bold"), bg="lightblue")
canvas.create_window((512, 30), window=title_label, anchor="n")


button_style = {"font": ("Helvetica", 20, "bold"), "bg": "#4682B4", "fg": "white", "activebackground": "#5A9BD4"}

open_image_button = Button(button_frame, text="Open Image", width=20, height=5,
                           command=create_button_functionality(open_image, "Open Image"), **button_style)
open_image_button.grid(row=0, column=0, padx=20, pady=10)

live_button = Button(button_frame, text="Live Detection", width=20, height=5,
                     command=create_button_functionality(live_detection, "Live Detection"), **button_style)
live_button.grid(row=0, column=1, padx=20, pady=10)

load_video_button = Button(button_frame, text="Load Video", width=20, height=5,
                           command=create_button_functionality(load_video, "Load Video"), **button_style)
load_video_button.grid(row=1, column=0, padx=20, pady=10)

close_button = Button(button_frame, text="Close", width=20, height=5,
                      command=create_button_functionality(root.quit, "Close"), **button_style)
close_button.grid(row=1, column=1, padx=20, pady=10)

button_order = [
    open_image_button,
    live_button,
    load_video_button,
    close_button
]

# index of the button currently selected
current_index = 0

# function which announces the current button the user has selected
def update_focus():
    button = button_order[current_index]
    button.focus_set()
    engine.say(button.cget("text"))
    engine.runAndWait()

# function for navigation between buttons using the arrows
def navigate(event):
    global current_index
    if event.keysym == "Up":
        current_index = (current_index - 1) % len(button_order)
    elif event.keysym == "Down":
        current_index = (current_index + 1) % len(button_order)
    elif event.keysym == "Left":
        current_index = (current_index - 1) % len(button_order)
    elif event.keysym == "Right":
        current_index = (current_index + 1) % len(button_order)
    update_focus()

# function to select the button
def select_button(event):
    global current_button
    button = button_order[current_index]
    if current_button != button.cget("text"):
        engine.say(f"{button.cget('text')} selected, press again to confirm.")
        engine.runAndWait()
        current_button = button.cget("text")
    else:
        button.invoke()
        current_button = None


root.bind("<Up>", navigate)
root.bind("<Down>", navigate)
root.bind("<Left>", navigate)
root.bind("<Right>", navigate)
root.bind("<Return>", select_button)


canvas.create_window((512, 100), window=button_frame, anchor="n")


video_label = Label(canvas, bg="lightblue")
video_label_window = canvas.create_window((512, 300), window=video_label, anchor="n")

# associating the press of R with the announcement repeat of prediciton
root.bind('<r>', repeat_announcement)

# associating the press of Q with the returning to the main page
root.bind('<q>', stop_running)

# vocal message displayed at the laungh of application
def play_welcome_message():
    engine.say("Welcome to Insight Eye, please choose an option to continue. To select an option, please use up and down arrows. After selecting an option, if you want to go back to the main page, please press Q.")
    engine.runAndWait()

root.after(1000, play_welcome_message)


root.mainloop()
