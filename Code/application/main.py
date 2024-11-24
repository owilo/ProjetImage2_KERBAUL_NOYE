import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.colorchooser import askcolor
import os
import subprocess

last_X = 0
last_Y = 0
firstDraw = True
lineColor = "#000000"
first_X, first_Y = 0, 0
current_X, current_Y = 0, 0

def enter_command(executable_name, ind):
    global output_file
    global bg_image_id
    print(ind)
    output_file = "../results/"+output_file_field.get()
    command = "../obscuration/"+executable_name+" "+input_file+" "+output_file+" "+str(first_X)+" "+str(first_Y)+" "+str(current_X)+" "+str(current_Y)

    for i in range(len(fields[ind])):
        if isinstance(fields[ind][i], tk.Radiobutton):
            command += " "+str(sensDistorsion)
            break
        elif not isinstance(fields[ind][i], tk.Label):
            command += " "+str(fields[ind][i].get())


    print(command)
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stderr)
        label_resultat.config(text=result.stderr)

        image = Image.open(output_file)
        image_tk = ImageTk.PhotoImage(image)

        if bg_image_id:
            canvas.delete(bg_image_id)
        canvas.image = image_tk
        bg_image_id = canvas.create_image(0, 0, anchor="nw", image=image_tk)

    except Exception as e:
        print(f"Erreur lors de l'exécution de la commande: {e}")
        label_resultat.config(text="Erreur d'exécution de la commande.")


def reset(event):
    global firstDraw
    firstDraw = True

def draw_line(event):
    global last_X
    global last_Y
    global firstDraw
    current_X = event.x
    current_Y = event.y
    if not firstDraw:
        canvas.create_line(last_X,last_Y,current_X,current_Y, fill=lineColor)
    else: 
        firstDraw = False
    last_X, last_Y = current_X, current_Y

def draw_rectangle(event):
    global first_X
    global first_Y
    global firstDraw
    global current_X
    global current_Y
    current_X = event.x
    current_Y = event.y
    if not firstDraw:
        canvas.delete("to_delete")
        canvas.create_line(first_X,first_Y,current_X,first_Y, fill=lineColor, tags="to_delete")
        canvas.create_line(first_X,first_Y,first_X,current_Y, fill=lineColor, tags="to_delete")
        canvas.create_line(current_X,current_Y,current_X,first_Y, fill=lineColor, tags="to_delete")
        canvas.create_line(current_X,current_Y,first_X,current_Y, fill=lineColor, tags="to_delete")
    else: 
        firstDraw = False
        first_X, first_Y = current_X, current_Y

def openFileExplorer():
    global bg_image_id, input_file
    image_file = filedialog.askopenfilename(
        title="Choisir une image",
        filetypes=[("Tous les fichiers", "*.*")]
    )

    if image_file:
        print("Fichier sélectionné : " + image_file)
        input_file = image_file

        image = Image.open(image_file)
        image_tk = ImageTk.PhotoImage(image)

        if bg_image_id:
            canvas.delete(bg_image_id)

        canvas.image = image_tk
        canvas.config(width=image.width, height=image.height)
        bg_image_id = canvas.create_image(0, 0, anchor="nw", image=image_tk)
        canvas.grid(row=10,column=4,pady=40)

def openColorDialog():
    global lineColor
    lineColor = askcolor(title="Changer de couleur")[1]

def closeWindow(event=None):
    window.destroy()

window = tk.Tk()
window.attributes("-fullscreen", True)
window.title("Application Projet Image M2")
window.bind("<Escape>",closeWindow)

bg_image_id = None

loadImageButton = tk.Button(window, text="Ouvrir une image", command=openFileExplorer)
loadImageButton.grid()

changeColorButton = tk.Button(window, text="Changer de couleur", command=openColorDialog)
changeColorButton.grid()

canvas = tk.Canvas(window)
canvas.bind("<B1-Motion>", draw_rectangle)
canvas.bind("<ButtonRelease-1>", reset)

input_file = ""
output_file = ""

boutons_obscuration = []
cpp_directory =  "../obscuration/"
cpp_files = os.listdir(cpp_directory)
cpp_files = [f for f in cpp_files if f.endswith(".cpp") and os.path.isfile(os.path.join(cpp_directory, f))]

output_label = tk.Label(window,text="Fichier de sortie : ")
output_label.grid(padx=20)
output_file_field = tk.Entry(window, width=30)
output_file_field.grid(padx=20)

label_resultat = tk.Label(window, text="", font=("Arial", 10, "italic"), fg="red")
label_resultat.grid()

sensDistorsion = 0

# print(cpp_files)

fields = [[tk.Label(window, text="Clé de chiffrement : ", relief="solid", width=30),tk.Entry(window, width=30)],
          [tk.Label(window, text="Clé de chiffrement : ", relief="solid", width=30),tk.Entry(window, width=30), tk.Label(window, text="Nombre de bits chiffrés : ", relief="solid", width=30),tk.Scale(window,from_=1, to=8, orient="horizontal", width=30)],
          [tk.Label(window, text="Décalage R : ", relief="solid", width=30),tk.Scale(window,from_=-10, to=10, orient="horizontal", width=30),tk.Label(window, text="Décalage G : ", relief="solid", width=30),tk.Scale(window,from_=-10, to=10, orient="horizontal", width=30),tk.Label(window, text="Décalage B : ", relief="solid", width=30),tk.Scale(window,from_=-10, to=10, orient="horizontal", width=30)],
          [tk.Label(window, text="Amplitude : ", relief="solid", width=30),tk.Scale(window,from_=1, to=20, orient="horizontal", width=30),tk.Label(window, text="Fréquence : ", relief="solid", width=30),tk.Scale(window,from_=0.1, to=1.0, resolution=0.05, orient="horizontal", width=30),tk.Label(window, text="Sens distorsion : ", relief="solid", width=30), tk.Radiobutton(window, text="Verticale", variable=sensDistorsion, value=0, width=30), tk.Radiobutton(window, text="Horizontale", variable=sensDistorsion, value=1, width=30) ],
          [tk.Label(window, text="Taille filtre : ", relief="solid", width=30),tk.Scale(window,from_=3, to=25, resolution=2, orient="horizontal")],
          [tk.Label(window, text="R : ", relief="solid", width=30),tk.Scale(window,from_=0, to=255, orient="horizontal"),tk.Label(window, text="G : ", relief="solid", width=30),tk.Scale(window,from_=0, to=255, orient="horizontal"),tk.Label(window, text="B : ", relief="solid", width=30),tk.Scale(window,from_=0, to=255, orient="horizontal")],
          [tk.Label(window, text="Taille bloc pixel : ", relief="solid", width=30),tk.Scale(window,from_=2, to=32, resolution=1, orient="horizontal")]]

for i in range(len(cpp_files)):
    button = tk.Button(window, text=cpp_files[i],command=lambda p=cpp_files[i].split('.')[0],ind=i: enter_command(p,ind))
    button.grid(row=0,column=i+1, padx=20,pady=20)
    for j in range(len(fields[i])):
        fields[i][j].grid(row=j+1,column=i+1, padx=10,pady=10)

window.mainloop()