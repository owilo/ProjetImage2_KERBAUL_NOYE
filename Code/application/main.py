import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.colorchooser import askcolor
import os
import subprocess
import cv2

last_X = 0
last_Y = 0
firstDraw = True
lineColor = "#000000"
first_X, first_Y = 0, 0
current_X, current_Y = 0, 0
isFullScreen = False

app_mode = False # False pour mode classique, True pour évaluation par IA 
obscurationMode=-1 # -1 pour All, sinon de 0 à 6

def setOptionInFrame(ind):
    for b in range(len(frame_buttons_obscuration_app1.winfo_children())):
        current_button = frame_buttons_obscuration_app1.winfo_children()[b]
        if isinstance(current_button, tk.Button):
            if b == ind+1:
                current_button.config(bg="green")
            else:
                current_button.config(bg="#d9d9d9")

    for w in frame_options_obscuration.winfo_children():
        w.pack_forget()
    
    for j in range(len(fields[ind])):
        fields[ind][j].pack(pady=10)
    
    apply_obscuration = tk.Button(frame_options_obscuration, text="Appliquer l'obscuration",command=lambda p=cpp_files[ind].split('.')[0],ind_method=ind: enter_command(p,ind_method))
    apply_obscuration.pack(pady=40)
    

def enter_command(executable_name, ind_method):
    global output_file
    global bg_image_id
    global sensDistorsion

    file_type = input_file.split('.')[1]

    if file_type in ["jpg","png"]:
        output_file = "../results/"+output_file_field.get()
        command = "../obscuration/"+executable_name+" "+input_file+" "+output_file+" "+str(first_X)+" "+str(first_Y)+" "+str(current_X)+" "+str(current_Y)

        for i in range(len(fields[ind_method])):
            if isinstance(fields[ind_method][i], tk.Radiobutton):
                command += " "+str(sensDistorsion.get())
                break
            elif not isinstance(fields[ind_method][i], tk.Label):
                command += " "+str(fields[ind_method][i].get())


        print(command)
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stderr)
            label_resultat.config(fg="red")
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
        
        command = "../evaluation/evaluate"+" "+input_file+" "+output_file
        print(command)
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stderr)
            label_resultat.config(fg="black")
            label_resultat.config(text=result.stdout)
        except Exception as e:
            print(f"Erreur lors de l'exécution de la commande: {e}")
            label_resultat.config(text="Erreur d'exécution de la commande.")

    elif file_type in ["mp4"]:
        output_file = "../results/"+output_file_field.get()
        command = "python3 ../videos/main.py"+" "+input_file+" "+output_file+" ../obscuration/"+executable_name

        for i in range(len(fields[ind_method])):
            if isinstance(fields[ind_method][i], tk.Radiobutton):
                command += " "+str(sensDistorsion.get())
                break
            elif not isinstance(fields[ind_method][i], tk.Label):
                command += " "+str(fields[ind_method][i].get())

        print(command)

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stderr)
            label_resultat.config(fg="red")
            label_resultat.config(text=result.stderr)

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
    name_file = filedialog.askopenfilename(
        title="Choisir une image ou une vidéo",
        filetypes=[("Tous les fichiers", "*.*")]
    )

    if name_file:
        file_type = name_file.split('.')[1]
        if file_type in ["jpg","png"]:
            print("Image sélectionnée : " + name_file)
            input_file = name_file

            image = Image.open(name_file)
            image_tk = ImageTk.PhotoImage(image)

            if bg_image_id:
                canvas.delete(bg_image_id)

            canvas.image = image_tk
            canvas.config(width=image.width, height=image.height)
            bg_image_id = canvas.create_image(0, 0, anchor="nw", image=image_tk)
            canvas.grid(row=2,column=0,pady=40)
        elif file_type in ["mp4"]:
            print("Vidéo sélectionnée : " + name_file)
            input_file = name_file

            video_file = cv2.VideoCapture(name_file)
            if not video_file.isOpened():
                print("Erreur : Impossible d'ouvrir la vidéo.")
                exit()
            
            ret, frame = video_file.read()
            video_file.release()
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image_tk = ImageTk.PhotoImage(image)

            if bg_image_id:
                canvas.delete(bg_image_id)

            canvas.image = image_tk
            canvas.config(width=image.width, height=image.height)
            bg_image_id = canvas.create_image(0, 0, anchor="nw", image=image_tk)
            canvas.grid(row=2,column=0,pady=40)

def show_frame_classique():
    frame_obscuration.grid(row=0, column=0, padx=20, pady=20)
    frame_buttons_obscuration_app1.grid(row=0, column=1, padx=20, pady=20)
    frame_options_obscuration.grid(row=0, column=2, padx=20, pady=20)
    frame_image_input.grid(row=0, column=0, padx=20, pady=20)

def openColorDialog():
    global lineColor
    lineColor = askcolor(title="Changer de couleur")[1]

def closeWindow(event=None):
    window.destroy()

def resize_window(event=None):
    global isFullScreen
    isFullScreen = not isFullScreen
    window.attributes("-fullscreen", isFullScreen)

def hide_frame_classique():
    for frame in frame_obscuration.winfo_children():
        frame.grid_forget()
    frame_obscuration.grid_forget()

def show_frame_evaluationIA():
    frame_evaluationIA.grid(row=0, column=0, padx=20, pady=20)
    frame_buttons_obscuration_app2.grid(row=0, column=0, padx=20, pady=20)
    frame_choix_train_dataset.grid(row=0, column=1, padx=20,pady=20)
    frame_evaluate.grid(row=0, column=2,padx=20,pady=20)
    frame_resultat_evaluation.grid(row=1,column=0)


def hide_frame_evaluationIA():
    for frame in frame_evaluationIA.winfo_children():
        frame.grid_forget()
    frame_evaluationIA.grid_forget()

def switch_app(event=None):
    global app_mode
    if not app_mode:
        hide_frame_classique()
        show_frame_evaluationIA()
    else:
        hide_frame_evaluationIA()
        show_frame_classique()

    app_mode = not app_mode

def setObscurationMode(ind):
    global obscurationMode
    for b in range(len(frame_buttons_obscuration_app2.winfo_children())):
        current_button = frame_buttons_obscuration_app2.winfo_children()[b]
        if isinstance(current_button, tk.Button):
            if b == ind+1:
                current_button.config(bg="green")
                obscurationMode = ind
            else:
                current_button.config(bg="#d9d9d9")

def launch_evaluation():
    global canvas_accuracy
    global canvas_confusion
    global window
    global accuracy_bg
    global confusion_bg

    obscurationMethod = cpp_files[obscurationMode].split('.')[0]
    command = "python3 ../evaluation/evaluate-cifar-obs.py"+" "+obscurationMethod+" "+str(train_dataset_is_obscurated.get())+" "+str(isContextuel.get())
    print(command)

    try:
        # Lancer un terminal xterm
        process = subprocess.Popen(["xterm", "-e", command])
        process.wait()

        print("Evaluation terminée")

        canvas_accuracy.pack(side="left",padx=20,pady=20)
        canvas_confusion.pack(side="left",padx=20,pady=20)

        image_accuracy = Image.open("accuracy_over_epochs.png")
        image_confusion = Image.open("confusion_matrix.png")
        image_accuracy = image_accuracy.resize((400, 300), Image.ANTIALIAS)
        image_confusion = image_confusion.resize((500, 400), Image.ANTIALIAS)
        image_accuracy_tk = ImageTk.PhotoImage(image_accuracy)
        image_confusion_tk = ImageTk.PhotoImage(image_confusion)

        if accuracy_bg:
            canvas_accuracy.delete(accuracy_bg)
        if confusion_bg:
            canvas_confusion.delete(confusion_bg)

        canvas_accuracy.image = image_accuracy_tk
        accuracy_bg = canvas_accuracy.create_image(0, 0, anchor="nw", image=image_accuracy_tk)
        canvas_confusion.image = image_confusion_tk
        confusion_bg = canvas_confusion.create_image(0, 0, anchor="nw", image=image_confusion_tk)

    except Exception as e:
        print(f"Erreur lors de l'exécution de la commande: {e}")
        label_resultat.config(text="Erreur d'exécution de la commande.")


    

window = tk.Tk()
window.attributes("-fullscreen", isFullScreen)
window.title("Application Projet Image M2")
window.bind("<Escape>",closeWindow)
window.bind('<Control-slash>', resize_window) 
window.bind('<Return>', switch_app) 

bg_image_id = None

accuracy_bg = None
confusion_bg = None

input_file = ""
output_file = ""

boutons_obscuration = []
cpp_directory =  "../obscuration/"
cpp_files = os.listdir(cpp_directory)
cpp_files = [f for f in cpp_files if f.endswith(".cpp") and os.path.isfile(os.path.join(cpp_directory, f))]
cpp_files.sort()

sensDistorsion = tk.IntVar()
train_dataset_is_obscurated = tk.IntVar()
isContextuel = tk.IntVar()

# Mode classique
##############################################################################################################
frame_obscuration = tk.Frame(window)

frame_buttons_obscuration_app1 = tk.Frame(frame_obscuration)

frame_options_obscuration = tk.Frame(frame_obscuration)

frame_image_input = tk.Frame(frame_obscuration)

show_frame_classique()

length_for_button = len(max(cpp_files, key=len))

loadImageButton = tk.Button(frame_image_input, width=40, text="Ouvrir une image ou une vidéo", command=openFileExplorer)
loadImageButton.grid(row=0, column=0, padx=20, pady=20)

changeColorButton = tk.Button(frame_image_input, width=40, text="Changer de couleur", command=openColorDialog)
changeColorButton.grid(row=1, column=0, padx=20, pady=20)

canvas = tk.Canvas(frame_image_input)
canvas.bind("<B1-Motion>", draw_rectangle)
canvas.bind("<ButtonRelease-1>", reset)

output_label = tk.Label(frame_image_input,width=40, text="Fichier de sortie : ")
output_label.grid(row=3, column=0, padx=20, pady=20)
output_file_field = tk.Entry(frame_image_input, width=40)
output_file_field.grid(row=4, column=0, padx=20, pady=20)

label_resultat = tk.Label(frame_image_input, text="", font=("Arial", 10, "italic"))
label_resultat.grid(row=5, column=0)

#print(cpp_files)

fields = [
          [tk.Label(frame_options_obscuration, text="Clé de chiffrement : ", relief="solid", width=30),tk.Entry(frame_options_obscuration, width=30)],
          [tk.Label(frame_options_obscuration, text="Clé de chiffrement : ", relief="solid", width=30),tk.Entry(frame_options_obscuration, width=30), tk.Label(frame_options_obscuration, text="Nombre de bits chiffrés : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=1, to=8, orient="horizontal", width=30)],
          [tk.Label(frame_options_obscuration, text="Décalage R : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=-10, to=10, orient="horizontal", width=30),tk.Label(frame_options_obscuration, text="Décalage G : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=-10, to=10, orient="horizontal", width=30),tk.Label(frame_options_obscuration, text="Décalage B : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=-10, to=10, orient="horizontal", width=30)],
          [tk.Label(frame_options_obscuration, text="Amplitude : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=1, to=20, orient="horizontal", width=30),tk.Label(frame_options_obscuration, text="Fréquence : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=0.1, to=1.0, resolution=0.05, orient="horizontal", width=30),tk.Label(frame_options_obscuration, text="Sens distorsion : ", relief="solid", width=30), tk.Radiobutton(frame_options_obscuration, text="Verticale", variable=sensDistorsion, value=0, width=30), tk.Radiobutton(frame_options_obscuration, text="Horizontale", variable=sensDistorsion, value=1, width=30) ],
          [tk.Label(frame_options_obscuration, text="Taille filtre : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=3, to=25, resolution=2, orient="horizontal")],
          [tk.Label(frame_options_obscuration, text="R : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=0, to=255, orient="horizontal"),tk.Label(frame_options_obscuration, text="G : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=0, to=255, orient="horizontal"),tk.Label(frame_options_obscuration, text="B : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=0, to=255, orient="horizontal")],
          [tk.Label(frame_options_obscuration, text="Taille bloc pixel : ", relief="solid", width=30),tk.Scale(frame_options_obscuration,from_=2, to=32, resolution=1, orient="horizontal")]]

label_obscuration = tk.Label(frame_buttons_obscuration_app1, text="Méthodes d'obscurations", relief="solid", width=30)
label_obscuration.pack(pady=10)
for i in range(len(cpp_files)):
    button = tk.Button(frame_buttons_obscuration_app1, width=length_for_button, text=cpp_files[i],command=lambda ind=i: setOptionInFrame(ind))
    button.pack(pady=10)
##############################################################################################################






# Evaluation par IA
##############################################################################################################
frame_evaluationIA = tk.Frame(window)
frame_buttons_obscuration_app2 = tk.Frame(frame_evaluationIA)

label_obscuration_app_IA = tk.Label(frame_buttons_obscuration_app2, text="Méthodes d'obscurations", relief="solid", width=30)
label_obscuration_app_IA.pack(pady=10)
for i in range(len(cpp_files)):
    button = tk.Button(frame_buttons_obscuration_app2, width=length_for_button, text=cpp_files[i], command=lambda ind=i: setObscurationMode(ind))
    button.pack(pady=10)


frame_choix_train_dataset = tk.Frame(frame_evaluationIA)
label_train_dataset = tk.Label(frame_choix_train_dataset, text="Obscurcir le dataset d'entraînement ?")
train_dataset_option1 = tk.Radiobutton(frame_choix_train_dataset, text="Oui", variable=train_dataset_is_obscurated, value=0, width=30)
train_dataset_option2 = tk.Radiobutton(frame_choix_train_dataset, text="Non", variable=train_dataset_is_obscurated, value=1, width=30)
label_train_dataset.pack(pady=10)
train_dataset_option1.pack(pady=10)
train_dataset_option2.pack(pady=(10,50))

label_contextuel = tk.Label(frame_choix_train_dataset, text="Contextuel ou Non-Contextuel ?")
contextuel_option1 = tk.Radiobutton(frame_choix_train_dataset, text="Contextuel (16x16 pixels)", variable=isContextuel, value=16, width=30)
contextuel_option2 = tk.Radiobutton(frame_choix_train_dataset, text="Non-Contextuel (32x32 pixels)", variable=isContextuel, value=32, width=30)
label_contextuel.pack(pady=10)
contextuel_option1.pack(pady=10)
contextuel_option2.pack(pady=10)

frame_evaluate = tk.Frame(frame_evaluationIA)
evaluate_button = tk.Button(frame_evaluate, text="Lancer l'évaluation",width=30, command=launch_evaluation)
evaluate_button.pack(pady=10)

frame_resultat_evaluation = tk.Frame(frame_evaluationIA)
canvas_accuracy = tk.Canvas(frame_resultat_evaluation)
canvas_confusion = tk.Canvas(frame_resultat_evaluation)

##############################################################################################################



window.mainloop()