import xml.etree.ElementTree as ET


class Stations():

    def __init__(self, path_xml_file):
        
        self._tree = ET.parse(path_xml_file)
        self._root = self._tree.getroot()

        self.stations = []

        self.get_stations()
        # self.print_stations()
        

    def get_stations(self):

        self.stations = []

        # Get station segments from .xml file
        # stations_root = self._root.find('Stations')

        for segment in self._root.findall('Station'):
            name = segment.find('Name').text
            position = float(segment.find('Position').text)
            
            self.stations.append([name, position])

    def print_stations(self):

       for station in self.stations:
           print('==')
           print('Name     : ' + station[0])
           print('Position : ' + str(station[1]))