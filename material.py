class acousticMaterial:
    def __init__(self, density, wave_speed):
        self.density = density
        self.wave_speed = wave_speed

    @property
    def acoustic_impedance(self):
        """Calculate and return the acoustic impedance of the material."""
        return self.density * self.wave_speed
    
    def bulk_modulus(self):
        """Calculate and return the bulk modulus of the material."""
        return self.density * self.wave_speed ** 2
