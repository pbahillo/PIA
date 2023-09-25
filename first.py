import model as m

def create_function():
    persona=m.Persona()
    pass

def read_function():
    pass

def update_function():
    pass

def delete_function():
    pass


out=False

while out!=True:
    command=input("Selecciona opcion:\n"
                  "1. Create\n"
                  "2. Read\n"
                  "3. Update\n"
                  "4. Delete\n"
                  "0.EXIT\n")
    match command:
        case "1":
            print ("Seleccionoado 1")
            create_function()
        case "2":
            print ("Seleccionado 2")
            read_function()
        case "3":
            print ("Seleccionoado 3")
            update_function()
        case "4":    
            delete_function()
            print ("Seleccionoado 4")
        case "0":
            out=True
            print("Finishing")
        case other:
            print("Invalid option")


