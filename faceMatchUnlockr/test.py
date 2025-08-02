import tkinter as tk
import keyboard
import threading
from aloo import call,speak

pressed_keys = set()

def block_combos(event):
    pressed_keys.add(event.name)
    if len(pressed_keys) > 1:
        return False

def release_key(event):
    if event.name in pressed_keys:
        pressed_keys.remove(event.name)

for k in ['windows','alt','tab','f4','ctrl','shift']:
    keyboard.block_key(k)

keyboard.on_press(block_combos)
keyboard.on_release(release_key)

def unlock():
    keyboard.unhook_all()
    root.destroy()

root = tk.Tk()
width = root.winfo_screenwidth()
height = root.winfo_screenheight()

root.overrideredirect(True)
root.geometry(f"{width}x{height}+0+0")
root.configure(bg="black")
root.attributes("-topmost", True)

frame = tk.Frame(root, bg="black")
frame.pack(fill="both", expand=True)

label = tk.Label(frame, text="🔒 System Locked\nFace recognition in progress...", fg="white", bg="black", font=("Arial", 28))
label.place(relx=0.5, rely=0.5, anchor="center")

def block_event(event): return "break"
root.bind_all("<Key>", block_event)
root.bind_all("<Button>", block_event)


def check_unlock():
    value = call()
    if value:
        root.after(0, unlock)
threading.Thread(target=check_unlock, daemon=True).start()
        


root.mainloop()

speak("Welcome back Vashu The Great")
