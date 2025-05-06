import xml.etree.ElementTree as ET
import datetime
import pytz


class Journey():

    '''Extract Journey Data from XML File
    
    This class constructs a Journey Object and extracts the journey data from
    an XML file.

    Attributes
    ----------
    starting_point: list
        This list is given in the following format [Position, Planned Departure Time, Name of starting station]

    
    stopping_point: list
        This list is given in the following format [Position, Planned Arrival Time, Name of stopping station]

    Methods
    ----------
    get_journey():
        This method extracts journey information from the XML File

    print_journey():
        This method prints details of the journey 
    '''

    def __init__(self, path_xml_file):
        
        self._tree = ET.parse(path_xml_file)
        self._root = self._tree.getroot()

        self.starting_point = []
        self.stopping_point = []

        self.get_journey()
        

    def get_journey(self):

        self.starting_point = []
        self.stopping_point = []

        # Get nodes
        starting_root = self._root.find('StartingPoint')
        stopping_root = self._root.find('StoppingPoint')
        
        # Get starting point position and departure time
        self.starting_point.append(float(starting_root.find('Position').text))
        self.starting_point.append(int(starting_root.find('PlannedDepartureTime').text))
        self.starting_point.append(starting_root.find('Name').text)

        # Get stopping point position and arrival time
        self.stopping_point.append(float(stopping_root.find('Position').text))
        self.stopping_point.append(int(stopping_root.find('PlannedArrivalTime').text))
        self.stopping_point.append(stopping_root.find('Name').text)

    def print_journey(self):

        print('=============================')
        print('        Journey Data         ')
        print('=============================')

        now = datetime.datetime.now(pytz.timezone('Europe/Berlin'))
        departure_time = now.strftime("%H:%M:%S")
        print('Starting Point   : ' + self.starting_point[2])
        print('     Position    : {} m'.format(self.starting_point[0]))
        print('     Departure   : {}'.format(departure_time))

        duration = self.stopping_point[1] - self.starting_point[1]
        print('Duration         : ' + str(duration) + ' minutes')

        arrival_time = (now + datetime.timedelta(minutes=duration)).strftime("%H:%M:%S")
        print('Stopping Point   : ' + self.stopping_point[2])
        print('     Position    : {} m'.format(self.stopping_point[0]))
        print('     Arrival     : {}'.format(arrival_time))

        print('=============================')