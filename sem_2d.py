# sem_2d.py

class Example2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def display(self):
        print(f"Coordinates: ({self.x}, {self.y})")

def main():
    # Create an instance of Example2D
    example = Example2D(5, 10)
    
    # Display the coordinates
    example.display()

if __name__ == "__main__":
    main()