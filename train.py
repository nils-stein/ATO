import xml.etree.ElementTree as ET


class Train():

    '''Extract Train Parameters fom XML File
    
    This class constructs the train object.
    It reads the train parameters from the XML File and also keeps track of the tarin odomotry data.

    Attributes
    ----------
    speed: float
    acceleration: float
    position: float
    parameters: Dict
        A dictionary of the train parameters

    Methods
    ----------
    get_parameters():
        Read all train parameters from the XML file

    print_parameters():
        Print all the train parameters
    '''

    def __init__(self, path_xml_file):

        self._tree = ET.parse(path_xml_file)
        self._root = self._tree.getroot()

        # Dynamic Train Values
        self.speed = 0
        self.acceleration = 0
        self.position = 0
        self.parameters = {}

        # Update all parameters call
        self.get_parameters()


    def get_parameters(self):

        self.parameters = {}
        textual_tags = {'BaseTrainType', 'TrainClassId', 'Caption', 
                            'IsAffectedByPowerRestriction', 'SupportsCoasting', 
                            'IsLoco'}

        for element in self._root:

            if element.tag in textual_tags:
                self.parameters[element.tag] = element.text
            else:
                self.parameters[element.tag] = float(element.text)


    def print_parameters(self):

        for pa in self.parameters:
            print(pa + " = " + str(self.parameters[pa]))
    
