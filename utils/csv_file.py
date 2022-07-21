import csv

class CSV:

    def __init__(self, path, header):
        """
        Class constructor
        :param path: Path to the file
        :param header: File header
        """
        self.file = open(path, mode='w', newline='')
        self.writer=csv.writer(self.file, delimiter=';')
        self.writer.writerow(header)

    def writerow(self, element):
        """
        Write given list to csv file, after operaion automaticly flushes buffer
        :param element: List of elelemnts
        """
        self.writer.writerow(element)
        self.file.flush()

    def __del__(self):
        """
        Class destructor, closing file
        """
        self.file.close()