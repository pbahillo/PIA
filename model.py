class Persona:
    nombre=""
    dni=""
    

    def __init__(self, nombre, dni):
        self.nombre=nombre
        self.dni=dni

    def get_nombre(self):
        return self.nombre
    def set_nombre(self,nombre):
        self.nombre=nombre

    def get_dni(self):
        return self.dni
    def set_dni(self,dni):
        self.dni=dni



def create_presona(nombre, dni):
    persona={
        "nombre":nombre,
        "dni":dni
    }
    return persona