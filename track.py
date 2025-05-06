from pathlib import Path
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


class Track():

    '''Track Constructor 
    
    This class constructs a track object which describes the speed limit segments, the gradient along the track
    and the curves of the track

    Attributes
    ----------
    file_type: str
        This attribute gives the kind of file from which the track data is extracted. It can be 'csv' or 'xml' and is read automatically from the file

    gradient_segments: list of lists
        This is a list of lists, each list within the main list is of the format [Starting Position of segment, Stopping Position of segment, Gradient]
        
    speed_limit_segments: list of lists
        This is a list of lists, each list within the main list is of the format [Starting Position of segment, Stopping Position of segment, Speed Limit]

    curve_segments: list of lists
        This is a list of lists, each list within the main list is of the format [Starting Position of segment, Stopping Position of segment, Radius of Curve]

    Methods
    ----------
    get_gradient_segments():
    get_speed_limit_segments():
    get_curve_segment():
    visualize(): Plot graphs showing track data
    '''

    def __init__(self, path_file):

        '''Initialisation'''

        self.file_type = None
        
        # Get file type
        with open(path_file, "r") as f:

            check = f.read(1)
    
            if check == "<":
                self.file_type = 'xml'
            else:
                self.file_type = 'csv'

        assert self.file_type != None, \
            'Track file type is unknown or of unacceptable format.'

        if self.file_type == 'csv':
        
            self.dataframe = pd.read_csv(path_file, sep=';', encoding= 'unicode_escape')
            self.dataframe.drop(['Abkuerzung', 'RealeKilometer[km]', 'Weiche[-]', 'Leistungsaufnahme[-]', \
                'Tunnel[-]'], inplace=True, axis='columns')

            # print(self.dataframe)

        elif self.file_type == 'xml':

            self._tree = ET.parse(path_file)
            self._root = self._tree.getroot()

        self.gradient_segments = []
        self.speed_limit_segments = []
        self.curve_segments = []

        self.get_segments()
        # print(self.gradient_segments)
        # self.visualize()
        

    def get_gradient_segments(self):

        '''
        The track can either be described via an XML file or a CSV file. This method, will read gradient data of the track from the XML or CSV file
        depending on the file the user inputs
        '''

        # Get gradient segments from .csv file

        if self.file_type == 'csv':

            grad_segments = []
            df_gradients = self.dataframe[['Position[km]', 'Neigung[o/oo]']].copy()
            gradient = None  # placeholder
            gradient_segment = []

            for _, row in df_gradients.iterrows():

                if row['Neigung[o/oo]'] != gradient:

                    if len(gradient_segment) == 3:
                        gradient_segment[1] = round(row["Position[km]"]*1000, 2)
                        grad_segments.append(gradient_segment)
                        gradient_segment = []

                    gradient_segment.append(round(row["Position[km]"]*1000, 2))
                    gradient_segment.append(None)
                    gradient_segment.append(row["Neigung[o/oo]"])
                    gradient = row["Neigung[o/oo]"]
            
            last_segment = [grad_segments[-1][1], round((df_gradients["Position[km]"].values[-1])*1000,2), \
                df_gradients["Neigung[o/oo]"].values[-1]]
            grad_segments.append(last_segment)

            for i, segment in enumerate(grad_segments):

                if i not in [0, len(grad_segments) - 1]:

                    if segment[1] - segment[0] < 800:

                        diff_to_back = abs(grad_segments[i-1][2] - segment[2])
                        diff_to_front = abs(grad_segments[i+1][2] - segment[2])
                        segment[2] = grad_segments[i-1][2] if diff_to_front > diff_to_back else grad_segments[i+1][2]


            for index, segment in enumerate(grad_segments):
                if index > 0:
                    if abs(segment[2] - grad_segments[index-1][2]) < 1:
                        grad_segments[index][2] = grad_segments[index][2]                                                                               # mögliche Anpassung: „grad_segments[index][2] = grad_segments[index][2]“ zu „grad_segments[index][2] = grad_segments[index-1][2]“

            gradients = np.array([seg[2] for seg in grad_segments])
            indices = np.where(np.diff(gradients,prepend=np.nan))[0]

            for i, index in enumerate(indices):
                if index != indices[-1]:
                    self.gradient_segments.append([grad_segments[index][0], grad_segments[indices[i+1]-1][1], grad_segments[index][2]])

            self.gradient_segments.append(grad_segments[-1])

            # some buffer segment to avoid error
            self.gradient_segments.append([self.gradient_segments[-1][1], self.gradient_segments[-1][1] + 1000, self.gradient_segments[-1][2]])
            self.gradient_segments.append([self.gradient_segments[-1][1], self.gradient_segments[-1][1] + 200, self.gradient_segments[-1][2]])

            #print("Gradient Segments:", self.gradient_segments)                                                                                        # prints für Expertentfernung (um Gradient-Segments zu sehen)

        
        elif self.file_type == 'xml':

            # Get gradient segments from .xml file
            self.gradient_segments = []
            gradient_root = self._root.find('GradientSegments')

            for segment in gradient_root.findall('GradientSegment'):
                start_pos = float(segment.find('StartPosition').text)
                end_pos = float(segment.find('EndPosition').text)
                gradient = float(segment.find('Gradient').text)
                self.gradient_segments.append([start_pos, end_pos, gradient])


    def _kmph_to_mps(kmph):
        '''Trnasform speed from km per hr to meters per second'''
        return kmph * (5.0/18.0)


    def get_speed_limit_segments(self):
        '''
        The track can either be described via an XML file or a CSV file. This method, will read speed limit data from the XML or CSV file depending
        depending on the file the user inputs
        '''

        if self.file_type == 'csv':  # Get speed limit segments from .csv file

            self.speed_limit_segments = []

            df_speed_limits = self.dataframe[['Position[km]', 'MaximaleGeschwindigkeit[km/h]']].copy()

            speed_limit = None  # placeholder
            speed_limit_segment = []

            for _, row in df_speed_limits.iterrows():

                if row['MaximaleGeschwindigkeit[km/h]'] != speed_limit:

                    if len(speed_limit_segment) == 3:

                        speed_limit_segment[1] = round(row["Position[km]"]*1000, 2)
                        self.speed_limit_segments.append(speed_limit_segment)
                        
                        speed_limit_segment = []

                    speed_limit_segment.append(round(row["Position[km]"]*1000,2)) # Converted to meters
                    speed_limit_segment.append(None)
                    speed_limit_segment.append(round(row["MaximaleGeschwindigkeit[km/h]"]*(5/18),2)) # Converted to m/s

                    speed_limit = row["MaximaleGeschwindigkeit[km/h]"]
                
            last_segment = [self.speed_limit_segments[-1][1], round((df_speed_limits["Position[km]"].values[-1])*1000,2), \
                round((df_speed_limits["MaximaleGeschwindigkeit[km/h]"].values[-1])*(5/8),2)]

            self.speed_limit_segments.append(last_segment)

            # some buffer segment to avoid error
            self.speed_limit_segments.append([self.speed_limit_segments[-1][1], self.speed_limit_segments[-1][1] + 1000, self.speed_limit_segments[-1][2]])
            self.speed_limit_segments.append([self.speed_limit_segments[-1][1], self.speed_limit_segments[-1][1] + 200, self.speed_limit_segments[-1][2]])

            #print("Speed Limit Segments:", self.speed_limit_segments)                                                                                  # prints für Expertentfernung  (um Speedlimit-Segments zu sehen)

        elif self.file_type == 'xml':

            # Get speed limit segments from .xml file
            self.speed_limit_segments = []
            
            speed_limit_root = self._root.find('SpeedLimitSegments')

            for segment in speed_limit_root.findall('SpeedLimitSegment'):
                start_pos = float(segment.find('StartPosition').text)
                end_pos = float(segment.find('EndPosition').text)
                speed_limit = float(segment.find('SpeedLimit').text)

                self.speed_limit_segments.append([start_pos, end_pos, speed_limit])


    def get_curve_segments(self):
        '''
        This method will read curve data fo the tack from the XML or CSV file that the user inputs
        '''

        if self.file_type == 'csv':  # Read from CSV

            self.curve_segments = []
            df_curves = self.dataframe[['Position[km]', 'Radius[m]']].copy()

            radius = None  # placeholder
            curve_segment = []

            for _, row in df_curves.iterrows():

                if row['Radius[m]'] != radius:

                    if len(curve_segment) == 3:

                        curve_segment[1] = round(row["Position[km]"]*1000, 2)
                        self.curve_segments.append(curve_segment)
                        curve_segment = []

                    curve_segment.append(round(row["Position[km]"]*1000,2)) # Converted to meters
                    curve_segment.append(None)
                    curve_segment.append(row["Radius[m]"]) 
                    radius = row["Radius[m]"]
                
            last_segment = [self.curve_segments[-1][1], round((df_curves["Position[km]"].values[-1])*1000,2), \
                df_curves["Radius[m]"].values[-1]]

            self.curve_segments.append(last_segment)

        elif self.file_type == 'xml':  # Read from XML

            self.curve_segments = []
            curve_root = self._root.find('CurveSegments')

            for segment in curve_root.findall('CurveSegment'):
                start_pos = float(segment.find('StartPosition').text)
                end_pos = float(segment.find('EndPosition').text)
                radius = float(segment.find('Radius').text)
                self.curve_segments.append([start_pos, end_pos, radius])


    def get_segments(self):

        # update all track parameters
        self.get_gradient_segments()
        self.get_speed_limit_segments()
        self.get_curve_segments()


    def visualize(self):

        # visualize track details
        import matplotlib.pyplot as plt

        arr_station_positions = [136, 3419, 5575, 13211, 22197, 27363, 31994, 36622, \
                             42082, 46110, 50406, 57974, 61297, 67003, 81993, 93975]
        
        station_positions = [round(x/1000, 1) for x in arr_station_positions]
        
        station_names = ['Stuttgart','Bad Cannstatt','Untertuerkheim','Esslingen', \
                         'Plochingen','Reichenbach','Ebersbach','Uhingen','Goeppingen', \
                         'Eislingen','Sueßen','Geislingen West','Geislingen','Amstetten', \
                         'Beimerstetten','Ulm Hbf']

        fig, axs = plt.subplots(3)
        fig.set_figwidth(10)
        fig.set_figheight(2*3)

        x_lim_low = min(
            self.gradient_segments[0][0],
            self.speed_limit_segments[0][0]
        )/1000
        x_lim_high = max(
            self.gradient_segments[-1][1],
            self.speed_limit_segments[-1][1]
        )/1000

        # Plot Speed Limits
        # axs[0].set_title('Speed Limit over Distance')
        axs[0].set_xlim([x_lim_low, x_lim_high])
        axs[0].set_xticks(station_positions)
        axs[0].step([segment[0]/1000 for segment in self.speed_limit_segments] + [self.speed_limit_segments[-1][1]/1000],
                    [segment[2] for segment in self.speed_limit_segments] + [self.speed_limit_segments[-1][2]],
                    where='post', color='r')
        axs[0].set_ylabel('Speed Limit (m/s)', labelpad=25)
        axs[0].set_xticklabels([])
        axs[0].grid(True)

        # Plot Gradients
        # axs[2].set_title('Curve over Distance')
        axs[1].set_xlim([x_lim_low, x_lim_high])
        axs[1].set_xticks(station_positions)
        axs[1].step([segment[0]/1000 for segment in self.curve_segments] + [self.curve_segments[-1][1]/1000],
                    [segment[2]/10000 for segment in self.curve_segments] + [self.curve_segments[-1][2]/10000],
                    where='post', color='g')
        axs[1].set_ylabel('Radius (x10^4 m)', labelpad=20)
        axs[1].set_xticklabels([])
        axs[1].grid(True)

        # Plot Speed Limits
        # axs[1].set_title('Gradient over Distance')
        axs[2].set_xlim([x_lim_low, x_lim_high])
        axs[2].set_xticks(station_positions)
        axs[2].step([segment[0]/1000 for segment in self.gradient_segments] + [self.gradient_segments[-1][1]/1000],
                    [segment[2] for segment in self.gradient_segments] + [self.gradient_segments[-1][2]],
                    where='post', color='b')
        axs[2].set_ylabel('Slope (\u2030)', labelpad=15)
        axs[2].set_xlabel('Distance (km)')
        plt.xticks(rotation='vertical')
        axs[2].grid()

        ax = axs[2].twiny()
        ax.set_xticks(station_positions, rotation=45)
        ax.set_xticklabels(station_names, rotation=45)
        ax.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
        ax.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
        ax.spines['bottom'].set_position(('outward', 46))
        ax.set_xlabel('Stations')
        ax.set_xlim(axs[2].get_xlim())
        
        plt.xticks(rotation='vertical')
        plt.show()


