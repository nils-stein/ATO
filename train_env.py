import math
import random
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
import pandas as pd
import ast

from sys import exit
import time
import numpy as np

from gym_train.envs.track import Track
from gym_train.envs.train import Train
from gym_train.envs.journey import Journey
from gym_train.envs.stations import Stations

from pathlib import Path

file_folder = Path("files/")
track = file_folder / "Stuggi_Ulm.csv"
train = file_folder / "train.xml"
journey = file_folder / "journey.xml"
stations = file_folder / "stations.xml"
sequential = True                                                                                                                                       # Expertentfernung: auf True gesetzt


class TrainEnv(gym.Env):

    '''
    Train Environment

    This class constructs the Environment in which the train will learn and also be evaluated.

    '''

    metadata = {
        'render_modes': ['human'],
        'render_fps': 30
    }
    

    def __init__(self, path_train=train, path_track=track, path_journey=journey, path_stations=stations, sequential_state=sequential):

        self.path_train = path_train # path to file describing train
        self.path_track = path_track # path to file describing track
        self.path_journey = path_journey # path to file describing journey

        # get train, track and journey information from .xml or .csv files
        self.train = Train(path_train) # train object
        self.num_powered_cars = 2 # number of powered units the train has
        self.track = Track(path_track) # track object
        self.journey = Journey(path_journey) # journey object
        self.stations = Stations(path_stations) # stations object
        self.df_journeys = pd.DataFrame(columns=['StartStationName', 'StartStationPosition', 'EndStationName', \
                                   'EndStationPosition', 'MinJourneyTime', 'PlannedJourneyTime', 'JourneySpeedLimitSegments', 'EstimatedSpeedPerSegment']) # dataframe for saving journey specs after journey time distribution algorithm
        self.read_journeys_from_excel() # read journey details for training from excel file

        # essential variables
        self.max_jerk = 0.0  # maximum jerk experienced
        self.E = 0.0  # energy used 
        self.distance_covered = 0.0  # distance covered by train on the current journey
        self.distance_to_destination = 0.0  # distance left to the destination
        self.cummulated_energy = 0.0  # cummulative energy consumed over the journey

        # MTD variables
        self.max_traction = 100  # maximum relative traction request the agent can choose
        self.max_braking = -100  # maximum relative braking request the agent can choose
        self.delta_position = 10  # distance between iterations of metrics saved while computing the journey time distribution algo
        self.journey_speed_limit_segments = []  # saves speed limits of journey

        self.minimal_time_profile_max_traction = []
        self.minimal_time_profile_max_braking = []
        self.minimal_time_profile = []
        self.estimated_velocity_limit_segments = []
        self.estimated_speed_profile = []
        self.res_time_segments = []
        self.min_res_time = 0

        # estimated speed
        self.v_est_current = 0.0  # current estimated speed on track
        self.v_est_next = 0.0  # next estimated speed on track
        
        # gradient details
        self.current_gradient = 0.0  # current gradient of track the train is running on
        self.next_gradient = 0.0  # the next gradient that the train will run
        self.distance_next_gradient = 0.0  #  distance to the gradient change

        # speed limit details
        self.current_speed_limit = 0.0  # current speed limit
        self.next_speed_limit = 0.0  # next speed limit the train will face
        self.distance_next_speed_limit = 0.0  # distance to the speed limit transition

        # init settings
        self.train.position = self.journey.starting_point[0]
        self.train.speed = 0.0
        self.initial_position = 0.0                                                                                                                     # ergänzt

        # forces
        self.F_tra = 0.0
        self.F_bra = 0.0
        self.F_gra = 0.0
        self.F_res = 0.0
        self.F_cur = 0.0
        self.F_tot = 0.0

        # time attributes
        self.journey_time = (self.journey.stopping_point[1] - self.journey.starting_point[1]) * 60  # the time planned for a journey in seconds
        self.accumulated_journey_time = 0  # the time the train has taken running from starting point to current position
        self.accumulated_action_time = 0  # time the action has been held constant 
        self.action_holding_time = 2.0  # time the agent should hold an action for
        self.delta_t = 0.1  # time between each dynamic simulation iteration of the train environemnet
        self.planned_time_left = self.journey_time  
        self.actual_time_left = self.journey_time
        self.delta_time_left = self.actual_time_left - self.planned_time_left  # the running punctuality error

        # future track state - these attributes are used when using a recurrent agent 
        self.sequential_state = sequential_state
        self.future_speed_limit_segments = []
        self.future_gradient_segments = []

        # reward attributes all initialised to 0.0
        self.reason_for_quit = 'None'
        self.r_energy = 0.0
        self.r_comfort = 0.0
        self.r_punctuality = 0.0
        self.r_safety = 0.0
        self.r_guide = 0.0
        self.total_reward = 0.0
        self.r_parking = 0.0                                                                                                                            # ergänzt für die Auswertungsfunktionen in main.py in der evaluation

        # rendering attributes
        self.rendering = False
        self.render_mode = 'human'
        self.screen = None
        self.surface = None
        self.screen_width = 1000
        self.screen_height = 400
        self.clock = None
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.blue = (0, 0, 255)
        self.lightblue = (173, 216, 230)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.grey = (150, 150, 150)
        self.lightgrey = (200, 200, 200)
        self.lightlightgrey = (230, 230, 230)
        self.speed_limit = 200
        self.render_gradient = 0
        self.max_speed_limit = 0
        self.render_speeds_arr = []
        self.render_positions_arr = []
        for segment in self.track.speed_limit_segments: # get max speed limit
            if self.max_speed_limit < segment[2]:
                self.max_speed_limit = segment[2]
        self.world_width = self.journey.stopping_point[0] - self.journey.starting_point[0]
        self.world_height_speed = self.max_speed_limit
        self.world_height_slope = 30 - (-30)  # Max/Min slope set at 30/-30 %
        self.scale_width = (self.screen_width - 100) / self.world_width
        self.scale_height_speed = (self.screen_height - 110) / self.world_height_speed
        self.scale_height_slope = (self.screen_height - 250) / self.world_height_slope

        # action and state spaces
        self.nA = 11 # number of actions
        #self.nF = 235 # number of features  # action, speed, distancetodestination, deltatimeleft, acceleration, jerk, distancetraveled, [speedlimits: 27 segmente], [gradients: 87 segmente] --> 7 + (27*2) + (87*2) = 235
        self.nF = 64 # number of features                                                                                                               # Anpassung zu ^, um auch gradients wie vor der Expertentfernung zu übergeben: 10 + (2*27) = 64

        self.action_space = spaces.Discrete(int(self.nA))

        self.observation_space = spaces.Dict({"1": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float64), 
                                              "speedlimits": spaces.Box(low=0, high=1, shape=(27,2), dtype=np.float64)})                                # Definieren des observation-space
        
        self.num_features = 64


    def _scale_state(self, obs):                                                                                                                        # angepasste Funktion

        '''
        This method takes in a state and scales it to feature values between 0 and 1.
        '''

        if not self.sequential_state:  # for non sequential states

            min = np.array([
                0.0,            # min gear
                0.0,            # min speed
                0.0,            # min distance left
                0.0,            # estimated speed
                -3600.0,        # min delta time left # experimentell: "-360" zu "-3600", damit Zeitfehler "ungekürzt" an agent gegeben wird
                0.0,            # min current speed limit
                0.0,            # min distance to next speed limit change
                0.0,            # min next speed limit
                -45.0,          # min gradient
                0.0,            # min distance to gradient change
                -45.0           # min next gradient
                ], 
                dtype=np.float32
            )

            max = np.array([
                10.0,           # max gear
                50.0,           # max speed
                100000,         # max distance left # experimentell: "2500" zu "100000", damit verbleibende Distanz "ungekürzt" an agent gegeben wird
                50.0,           # max v_est
                3600.0,         # max delta time left # experimentell: "360" zu "3600", damit Zeitfehler "ungekürzt" an agent gegeben wird
                50.0,           # max current speed limit
                2500,           # max distance to next speed limit change
                50.0,           # max next speed limit
                45.0,           # max gradient
                2500,           # max distance to gradient change
                45.0            # max next gradient
                ], 
                dtype=np.float32
            )

            # return 2 * np.divide(np.subtract(obs, min), np.subtract(max, min)) - 1 # scale to -1, 1
            return np.divide(np.subtract(obs, min), np.subtract(max, min))  # scale to 0, 1
        
        else:  # for sequential states                                                                                                                  # Expertentfernung: angepasst für neue Observation

            min_1 = np.array([
                0.0,            # min gear
                0.0,            # min speed
                -10.0,          # min distance left                                                                                                     # Anpassung, um den Bereich nach dem Endpunkt vom Bereich davor zu unterscheiden
                -1.2,           # min acceleration
                -3600.0,        # min time left
                0.0,            # min jerk
                -10.0,            # min distancetraveled
                -45.0,          # min gradient                                                                                                          # Anpassung, um gradients zu übergeben
                -10.0,            # min distance to gradient change                                                                                     # Anpassung, um gradients zu übergeben
                -45.0           # min next gradient                                                                                                     # Anpassung, um gradients zu übergeben
                ], 
                dtype=np.float64
            )

            max_1 = np.array([
                10.0,           # max gear
                50.0,           # max speed
                99990.0,        # max distance left                                                                                                     # angepasst, damit Gesamtskalierung 100000 bleibt
                1.2,            # max acceleration
                3600.0,         # max time left
                15.0,           # max jerk
                99990.0,       # max distancetraveled
                45.0,           # max gradient                                                                                                          # Anpassung, um gradients zu übergeben
                99990.0,         # max distance to gradient change                                                                                      # Anpassung, um gradients zu übergeben
                45.0            # max next gradient                                                                                                     # Anpassung, um gradients zu übergeben
                ], 
                dtype=np.float64
            )

            min_speed_limit_data = np.array([
                0.0,            # min speed_limit
                -10.0             # min speed limit segment length
                ], 
                dtype=np.float64
            )

            max_speed_limit_data = np.array([
                50.0,           # max speed limit
                99990.0        # max speed limit segment length
                ], 
                dtype=np.float64
            )


            scaled_obs_1 = np.array(np.divide(np.subtract(obs[0], min_1), np.subtract(max_1, min_1)), dtype=np.float64) # scale to 0 - 1

            obs_speed_limits = np.array(obs[1], dtype=np.float64)
            speed_limits_scaled = (obs_speed_limits[:, 0] - min_speed_limit_data[0]) / (max_speed_limit_data[0]- min_speed_limit_data[0])
            limit_lengths_scaled = (obs_speed_limits[:, 1] - min_speed_limit_data[1]) / (max_speed_limit_data[1]- min_speed_limit_data[1])
            scaled_obs_speed_limits = np.column_stack((speed_limits_scaled, limit_lengths_scaled))

            scaled_obs_dict = {"1": np.array(scaled_obs_1, dtype=np.float64), "speedlimits": np.array(scaled_obs_speed_limits, dtype=np.float64)}       # Erstellen des skalierten observation-dicts, das an den agent übergeben wird

            return (scaled_obs_dict)


    def get_estimated_speed(self, current_position):                                                                                                    # Expertentfernung: vorerst noch im Code, da v_est auch für Diagramme verwendet wird und zur optischen Abschätzung der Ergebnisse beiträgt

        '''This function will get the current and next estimated speeds of the train that were calculated by the journey time distribution algorithm'''

        assert len(self.journey_speed_limit_segments) > 0, \
            "No journey speed limit segments have been passed from the expert knowledge."
        
        # if train exceeds stopping point
        if self.train.position >= self.journey.stopping_point[0]:
            self.v_est_current = 0.0
            self.v_est_next = 0.0

        else:
            for index, segment in enumerate(self.journey_speed_limit_segments):
                if segment[0] <= current_position <= segment[1]:
                    self.v_est_current = self.estimated_velocity_limit_segments[index]
                    self.v_est_next = self.estimated_velocity_limit_segments[index+1]

    
    def get_gravitational_force(self, current_position):

        '''This function will calculate the gravitational forces that the train experiences'''

        front_position = current_position
        rear_position = current_position - self.train.parameters['Length']

        # find current gradient on track
        for index, segment in enumerate(self.track.gradient_segments):
            # locate front of train in gradient segments
            if segment[0] < front_position <= segment[1]:
                self.current_gradient = segment[2]
                self.render_gradient = segment[2]
                self.next_gradient = self.track.gradient_segments[index+1][2]
                self.distance_next_gradient = segment[1] - current_position
                # check if rear is in same gradient segment
                if segment[0] < rear_position <= segment[1]:
                    slope = segment[2]
                    return -1 * self.train.parameters['Mass'] * 9.81 * np.sin(np.arctan(slope/1000))
                # check if rear is still in previous gradient segment
                elif self.track.gradient_segments[index-1][0] < rear_position <= self.track.gradient_segments[index-1][1]:
                    slope_1 = segment[2]
                    slope_2 = self.track.gradient_segments[index-1][2]
                    slope_factor = (front_position-segment[0]) * np.sin(np.arctan(slope_1/1000)) + \
                        (segment[0]-rear_position) * np.sin(np.arctan(slope_2/1000))
                    return -1 * ((self.train.parameters['Mass'] / self.train.parameters['Length']) * 9.81 * slope_factor)
                # check if rear is still in gradient segment before previous segment
                elif self.track.gradient_segments[index-2][0] < rear_position <= self.track.gradient_segments[index-2][1]:
                    slope_1 = segment[2]
                    slope_2 = self.track.gradient_segments[index-1][2]
                    slope_3 = self.track.gradient_segments[index-2][2]
                    slope_factor = (front_position-segment[0]) * np.sin(np.arctan(slope_1/1000)) + \
                        (self.track.gradient_segments[index-1][1] - self.track.gradient_segments[index-1][0]) * np.sin(np.arctan(slope_2/1000)) + \
                        (self.track.gradient_segments[index-1][0] - rear_position) * np.sin(np.arctan(slope_3/1000))
                    return -1 * ((self.train.parameters['Mass'] / self.train.parameters['Length']) * 9.81 * slope_factor)
                else:
                    print("Segment: ", segment, "position: ", self.train.position)                                                                      # angepasst für mehr Infos im Zuge einer Fehlersuche
                    raise ValueError('Train length spans across more than 3 Gradient Segments')


    def get_curve_resistance(self, current_position):

        '''This function calculates the curve resistance that the train experiences depending on where on the track the train is currently'''

        car_center = current_position - (self.train.parameters['Length'] / 2)
        for _, segment in enumerate(self.track.curve_segments):
            if segment[0] < car_center <= segment[1]:
                if segment[2] != 0: 
                    if segment[2] >= 300:
                        return -1 * (self.train.parameters['Mass'] / 1000) * 9.81 * (500/(segment[2] - 30))
                    elif segment[2] < 300:
                        return -1 * (self.train.parameters['Mass'] / 1000) * 9.81 * (650/(segment[2] - 55))
                else:
                    return 0
                break
        return 0


    def get_drag_resisitance(self, current_speed):

        '''This function will calculate the drag forces that act on the train against its motion'''

        A = self.train.parameters['DragAmass'] * self.train.parameters['Mass'] + \
            self.train.parameters['DragAaxle'] * self.train.parameters['NumAxles']
        B = 0
        C = self.train.parameters['DragC'] + \
            self.train.parameters['DeltaDragC'] * self.train.parameters['NumCars']
        
        return -1 * (A + B * abs(current_speed) + C * (current_speed ** 2))


    def get_traction_force(self, action, current_speed):

        '''This function takes in the action the agent takes and the speed of the train and computes the traction force'''

        assert action > 0, "Action for traction can not be negative."
        # mu = 0.161 + (7.5 / (current_speed + 44))  # for dry track
        if np.round(current_speed, 3) != 0:
            F_tra = (action/100) * min(self.train.parameters['MaxForce'], (self.train.parameters['MaxPower'] / current_speed))
        else: 
            F_tra = (action/100) * self.train.parameters['MaxForce']

        return F_tra


    def get_braking_force(self, action):

        '''This function computes the braking force the train experiences depending on the action of the agent'''

        eff_Mass = self.train.parameters['Mass'] * self.train.parameters['RotatingMassesCoefficient']
        k = (self.num_powered_cars/self.train.parameters['NumCars'])
        lambda_a = abs(action)
        lambda_t = abs(action) #  self.train.parameters['BrakedWeightPercentageP']
        lambda_tot = (k*lambda_a + (1-k)*lambda_t)*(self.train.parameters['Mass']/eff_Mass)
        
        a_max_dec_SB = self.train.parameters['BrakedWeightToDecelerationRateLinear'] * lambda_tot + \
            self.train.parameters['BrakedWeightToDecelerationRateOffset']

        F_pref_brake = self.train.parameters['PrefferedDecelerationRate'] * eff_Mass
        F_max_SB = a_max_dec_SB * eff_Mass

        return -1 * min(F_max_SB, F_pref_brake)
            
    
    def acceleration(self, action, current_position, current_speed):

        '''This function gets the acceleration of the train'''

        self.F_gra = self.get_gravitational_force(current_position)     
        self.F_cur = self.get_curve_resistance(current_position)
        self.F_res = self.get_drag_resisitance(current_speed)
        if action > 0:
            self.F_tra = self.get_traction_force(action, current_speed)
            self.F_bra = 0
        elif action < 0:
            self.F_tra = 0
            self.F_bra = self.get_braking_force(action)
        else: 
            self.F_tra = 0
            self.F_bra = 0

        self.F_tot = self.F_tra + self.F_bra + self.F_gra + self.F_cur + self.F_res
        eff_Mass = self.train.parameters['Mass'] * self.train.parameters['RotatingMassesCoefficient']

        return self.F_tot / eff_Mass


    def get_speed_limit_data(self, current_position):

        '''This function gets the current speed limit, the next speed limit, and the distance the train has left to next speed limit'''

        for index, segment in enumerate(self.track.speed_limit_segments):
            if segment[0] < current_position <= segment[1]:
                self.current_speed_limit = segment[2]
                self.speed_limit = segment[2]
                self.next_speed_limit = self.track.speed_limit_segments[index+1][2]
                self.distance_next_speed_limit = segment[1] - current_position

                break


    def get_gradient_data(self, current_position):

        '''This function gets the current track gradient, the next track gradient and the distance the train has to the next track gradient'''

        for index, segment in enumerate(self.track.gradient_segments):
            if segment[0] < self.train.position <= segment[1]:
                self.current_gradient = segment[2]
                self.next_gradient = self.track.gradient_segments[index+1][2]
                self.distance_next_gradient = segment[1] - current_position

                break


    def get_future_track_data(self):                                                                                                                    # angepasste Funktion: Ermitteln der future_speed_limit_segments für die neue observation

        '''
        This function is only activated when we are using the recurrent neural network to learn policy
        '''

        self.future_speed_limit_segments.clear()
        speed_limit_segment_counter = 0
                
        for segment in self.track.speed_limit_segments:

            if segment[0] <= self.train.position < segment[1]:
                if self.journey.stopping_point[0] > segment[1]:
                    self.future_speed_limit_segments.append([segment[2], segment[1] - self.train.position])
                    speed_limit_segment_counter += 1
                else:
                    self.future_speed_limit_segments.append([segment[2], (self.journey.stopping_point[0]) - self.train.position])
                    speed_limit_segment_counter += 1

            elif segment[0] > self.train.position and segment[1] < self.journey.stopping_point[0]:
                self.future_speed_limit_segments.append([segment[2], segment[1] - segment[0]])
                speed_limit_segment_counter += 1

            elif segment[0] <= self.journey.stopping_point[0] <= segment[1]:
                self.future_speed_limit_segments.append([segment[2], (self.journey.stopping_point[0]) - segment[0]])
                speed_limit_segment_counter += 1

            if segment[0] > self.journey.stopping_point[0] and speed_limit_segment_counter < 27:                                                        # Segmente nach dem Ende der journey mit "Bremssegmenten" füllen
                self.future_speed_limit_segments.append([0, 1000])
                speed_limit_segment_counter += 1
            
        if speed_limit_segment_counter < 27:                                                                                                            # Sicherstellen, dass die Observation immer die gleiche Form hat -> unbelegte Segmente nach dem Ende der journey mit "Bremssegmenten" füllen
            while True:
                
                if speed_limit_segment_counter == 27:
                    break
                
                self.future_speed_limit_segments.append([0, 1000])

                speed_limit_segment_counter += 1
    
    
    def quit_episode(self):                                                                                                                             # angepasste Funktion

        '''This function checks terminal conditions of an episode'''
        
        if (self.train.speed < 0.0 and self.train.position > self.journey.starting_point[0] and self.train.position < 
            (self.journey.stopping_point[0] - 5)) or (self.train.speed < 0.0 and self.train.position <= self.journey.starting_point[0]):                # nicht nur Rückwärtsfahrt, sondern bereits Anhalten soll zu done führen, ausserdem: Zurückfallen hinter den Startpunkt ist nicht erlaubt
            self.reason_for_quit = 'TrainStopped'
            return True
        
        elif self.train.position >= (self.journey.stopping_point[0] - 5) and self.train.position <= (self.journey.stopping_point[0] + 5) and self.train.speed <= 0.0: # If train reaches destination    # angepasst, damit der Zug auch den Bereich erreichen kann, in dem er zu weit hinten parkt
            self.reason_for_quit = 'AtDestination'
            return True
        
        elif self.train.position > (self.journey.stopping_point[0] + 5):  # If train exceeds parking tolerance                                          # hinzugefügt, damit der Zug auch den Bereich erreichen kann, in dem er zu weit hinten parkt
            self.reason_for_quit = 'TooFar'
            return True
        
        # [IDEA] : Penalize when train runs too far behind.
        
        return False


    def reward_function(self, done):

        '''This function will calculate the reward that the agent receives for taking a certain action'''

        self.r_safety = 0.0
        self.r_parking = 0.0
        self.r_comfort = 0.0
        self.r_guide = 0.0
        self.r_punctuality = 0.0
        self.r_energy = 0.0
        k = 0.5                 # k \in [0,1], k -> 0 => Low influence of Energy Consumption on learning
        half_tol_parking = 5         # meters
        tol_punctuality = 60    # seconds

        ''' Reward Metrics '''
        eff_Mass = self.train.parameters['Mass'] * self.train.parameters['RotatingMassesCoefficient']
        max_acceleration = self.train.parameters['MaxForce'] / eff_Mass
        E_max = self.train.parameters['MaxSpeed'] * max_acceleration * self.action_holding_time
        self.E = (abs(self.F_tra + self.F_bra)/eff_Mass) * abs(self.train.speed) * self.action_holding_time
        self.cummulated_energy += self.E

        ''' 1. Safety - Per Step Reward '''
        #self.r_safety = -1.0 if (self.train.speed < 0.0 and self.distance_to_destination > 5) or self.train.speed > self.speed_limit or (self.train.speed < (self.speed_limit*0.5) and self.distance_to_destination > 350 and self.distance_covered > 250) else 0.0 # Experimentell: Angepasst um Parken im Parkbereich nicht zu bestrafen und "Schleichen", also eine Fahrt bei unter 50% des Geschwindigkeitslimits zu bestrafen (außer am Anfang und am Ende der Strecke)
        self.r_safety = -1.0 if self.train.speed > self.speed_limit else 0.0

        ''' 2. Comfort - Per Step Reward '''
        self.r_comfort = 0.0 if self.train.speed <= (5/3.6) or (self.max_jerk <= 4.0) else -1.0                                                         # Bedingung ergänzt: "self.train.speed <= (5/3.6) or"

        ''' 3. Energy - Per Step Reward ''' #Experimentell als terminal reward getestet
        if done:
         initial_distance = self.journey.stopping_point[0] - self.journey.starting_point[0]
         max_energy_expected = E_max * (initial_distance / 1000)
         energy_ratio = self.cummulated_energy / (max_energy_expected + 1e-5)
         self.r_energy = max(1.5 - energy_ratio, -3.0)

        #self.r_energy = 1.0 - k * (self.E / E_max)                                 #Belohnung pro step führt zu schleichen, da Agent viele kleine rewards sammelt

        ''' 4. Guide - Per Step Reward '''
        # if ((self.delta_time_left < (-tol_punctuality)) and self.r_safety == 0.0 and self.r_comfort == 0.0): # experimentell: Stufung um agent dazu zu bringen die zeit besser einzuhalten
        #     self.r_guide = 1.0
        # else:
        #     self.r_guide = 0.0
        self.r_guide = 1.0

        ''' 5. & 6. Parking & Punctuality - Terminal Reward '''
        if done:

            if self.reason_for_quit == 'TrainStopped':                                                                                                  # bei Anhalten mitten auf der Strecke:
                #parking_error = self.journey.stopping_point[0] - self.train.position           # verworfene experimentelle rewardverläufe ...
                #self.r_parking = (((-12) - (-8)) / ((self.journey.stopping_point[0]-self.initial_position) - 5)) * self.distance_to_destination + (-8 - ((-12 - -8) / ((self.journey.stopping_point[0]-self.initial_position) - 5)) * 5)
                #self.r_parking = 8*(1 - (abs(self.distance_to_destination) / half_tol_parking) ** 0.63)
                #self.r_punctuality = (-8 - abs(self.delta_time_left)/60)
                #self.r_punctuality = -8*((abs(self.delta_time_left) / tol_punctuality) ** 0.63)
                #self.r_punctuality = 8*(1 - (abs(self.delta_time_left) / tol_punctuality) ** 0.63)
                #self.r_parking = -1
                #self.r_parking = 2*(1 - (self.distance_to_destination/99985))                  # ...verworfene experimentelle rewardverläufe
                self.r_parking = -2-(2*(self.distance_to_destination/99985))
                #self.r_punctuality = -1                                                        #verworfen, da Bestrafung bei großer Verspätung zu gering
                scaling = 3.0
                self.r_punctuality = max(- scaling * (abs(self.delta_time_left) / tol_punctuality) ** 0.63, -8.0)

            elif self.reason_for_quit == 'AtDestination':                                                                                               # wenn im erwünschten Parkbereich angehalten (v <= 0) 
                self.r_parking = max(5.0 * (1 - (abs(self.distance_to_destination) / half_tol_parking)), 0.0)
                 # BONUS: Perfektes Parken (±1m) gibt zusätzlichen Reward
                if abs(self.distance_to_destination) <= 1.0:
                    self.r_parking += 2.0
                #self.r_parking = max(((2*(1 - (abs(self.distance_to_destination) / half_tol_parking) ** 0.63)))+2, -1)                                  # angepasst: "2 * (...)" ergänzt um Belohnung höher ausfallen zu lassen; reward nach Thesis-Funktion mit Anpassungen
                scaling = 3.0
                self.r_punctuality = max(- scaling * (abs(self.delta_time_left) / tol_punctuality) ** 0.63, -8.0)
                #self.r_punctuality = max(2*(1 - (abs(self.delta_time_left) / tol_punctuality) ** 0.63), -1.0)                                           # angepasst: "2 * (...)" ergänzt um Belohnung höher ausfallen zu lassen
                #self.r_punctuality = 8*(1 - (abs(self.delta_time_left) / tol_punctuality) ** 0.63) # verworfener experimenteller rewardverlauf

            elif self.reason_for_quit == 'TooFar': 
                self.r_parking = -15  # oder noch härter                                                                                                   # wenn maximale Parktoleranz nach hinten überschritten
                #self.r_parking = (0.5)*(((-0.08)*self.train.speed) - 8)                                  #verworfen, da somit das überfahren antrainiert wird                                               # negativer reward hoch um diesen Fall stark zu bestrafen und verlauf in Abhängigkeit von speed
                scaling = 3.0
                self.r_punctuality = max(- scaling * (abs(self.delta_time_left) / tol_punctuality) ** 0.63, -8.0)
                #self.r_punctuality = max((-6 - abs(self.delta_time_left))/60, -10)                    # verworfene experimentelle rewardverläufe ...
                #self.r_punctuality = -8*((abs(self.delta_time_left) / tol_punctuality) ** 0.63)
                #self.r_punctuality = 8*(1 - (abs(self.delta_time_left) / tol_punctuality) ** 0.63)    # ...verworfene experimentelle rewardverläufe
               # self.r_punctuality = -1
            

        ''' Total Reward '''                                                                                                                            # angepasst: safety und comfort verdoppelt
        self.total_reward = 0.2 * self.r_safety + \
                            0.2 * self.r_comfort + \
                            0.1 * self.r_guide * self.r_energy + \
                            0.5 * self.r_punctuality + \
                            1.0 * self.r_parking

        return self.total_reward
    

    def translator(self, action):
        '''Translate agent action \in [0, 11] to RTB_req \in [-100, 100]'''
        return (action - (self.nA-1)/2) * (100/((self.nA-1)/2))
    

    def step(self, action):

        '''This function takes an action and performs the simmulation of the train environment for the duration of the action holding time'''

        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid" # Check if action is valid
        translated_action = self.translator(action)  # translate action
        done = False
        self.max_jerk = 0.0
        #initial_position = self.train.position                                                                                                         # Auskommentiert um unten die insgesamt zurückgelegte Strecke zu ermitteln anstatt der in dem step zurückgelegten Strecke

        while not (done or self.accumulated_action_time > self.action_holding_time):

            old_speed = self.train.speed
            old_acceleration = self.train.acceleration

            # Train Acceleration
            self.train.acceleration = self.acceleration(translated_action, self.train.position, self.train.speed)

            # Train Speed
            delta_speed = self.train.acceleration * self.delta_t
            self.train.speed += delta_speed

            # Train Position
            delta_position = old_speed * self.delta_t + 0.5 * self.train.acceleration * (self.delta_t ** 2)
            self.train.position += delta_position

            # Train Jerk
            self.max_jerk = max(self.max_jerk, abs(self.train.acceleration - old_acceleration)/self.delta_t)
            
            # Get estimated speed and speed limit data
            self.get_estimated_speed(self.train.position)                                                                                               # Expertentfernung: vorerst noch im Code, da v_est auch für Diagramme verwendet wird und zur optischen Abschätzung der Ergebnisse beiträgt
            self.get_speed_limit_data(self.train.position)
            self.get_gradient_data(self.train.position)                                                                                                 # Ergänzt (zur Sauberkeit, gradient-Segmente werden eigentlich auch bei self.acceleration schon aktualisiert)
            done = self.quit_episode()
            
            if self.rendering: # Render train environment
                self.render_speeds_arr.append(self.train.speed)
                self.render_positions_arr.append(self.train.position)
                self.render()
                time.sleep(0.0001)

            self.accumulated_action_time += self.delta_t
        
        self.accumulated_journey_time += self.accumulated_action_time
        self.accumulated_action_time = 0
        self.delta_time_left = self.accumulated_journey_time - self.journey_time                                                                        # Expertentfernung: angepasst auf tatsächlichen Zeitfehler, unabhängig von JTDA
        self.get_future_track_data()                                                                                                                    # Expertentfernung: Ermitteln der future_speed_limit_segments

        self.distance_covered = self.train.position - self.initial_position                                                                             # Angepasst um die insgesamt zurückgelegte Strecke zu ermitteln anstatt der in dem step zurückgelegten Strecke
        self.distance_to_destination = self.journey.stopping_point[0] - self.train.position                                                             # neu angepasst gegenüber mit Expert um den Bereich nach dem Endpunkt vom Bereich davor zu unterscheiden

        # Get Reward
        reward = self.reward_function(done)

        if not self.sequential_state:

            self.observation = (
                    action,                                       # current action
                    self.train.speed,                             # current speed 
                    min(self.distance_to_destination, 100000),      # distance to next stopping point # experimentell: "2500" zu "100000", damit verbleibende Distanz "ungekürzt" an agent gegeben wird
                    self.v_est_current,                           # current estimated speed
                    min(self.delta_time_left, 3600),               # actual time to destination # experimentell: "300" zu "3600", damit Zeitfehler "ungekürzt" an agent gegeben wird
                    self.speed_limit,                             # current speed limit
                    min(self.distance_next_speed_limit, 2500),    # current speed limit length
                    self.next_speed_limit,                        # next speed limit
                    self.current_gradient,                        # current gradient
                    min(self.distance_next_gradient, 2500),       # current gradient length
                    self.next_gradient                            # next gradient
            )

            #if done: # verschoben: reset der obs auf 0 erst am Anfang der reset Funktion, damit letzter step noch mit normaler obs gemacht wird
            #    self.observation = [0.0 for _ in range(self.num_features)]

            return self._scale_state(np.array(self.observation, dtype=np.float32)), reward, done, {}
        
        else:                                                                                                                                           # Expertentfernung: angepasst auf neue Observation

            self.observation = (np.array([action, self.train.speed, min(self.distance_to_destination, 99990.0), self.train.acceleration,
                                          min(self.delta_time_left, 3600), min(self.max_jerk, 15), min(self.distance_covered, 99990.0), self.current_gradient, min(self.distance_next_gradient, 99990.0), self.next_gradient]), 
                                          np.array(self.future_speed_limit_segments))

            # if done:                                                                                                                                  # verschoben: reset der obs auf 0 erst am Anfang der reset Funktion, damit letzter step noch mit normaler obs gemacht wird
            #     self.observation = (
            #         [0.0] * 5,
            #         [[0.0, 0.0]],
            #         [[0.0, 0.0]]
            #     )

            return self._scale_state(self.observation), reward, done, {}


    def read_journeys_from_excel(self):

        '''Read training journeys from excel file'''

        self.df_journeys = pd.read_excel('files/journeys.xlsx')


    def get_all_possible_journeys(self):

        print('INFO: Getting all possible journeys between the stations along the track provided.')

        for index, station in enumerate(self.stations.stations):
            
            start_station = station
            i = index

            while i < len(self.stations.stations)-1:

                end_station = self.stations.stations[i+1]

                self.journey.starting_point[0] = start_station[1]
                self.journey.stopping_point[0] = end_station[1]

                self._minimal_speed_profile(self.journey.starting_point[0], 0.0)
                min_res_times = self._minimal_time_distribution()
                self.journey_time = 1.2 * self.min_res_time
                self.journey.stopping_point[1] = np.round(self.journey_time / 60, 2) # in minutes
                self._fitted_time_distribution(min_res_times)

                limits = []
                estspeeds = []

                for limit in self.journey_speed_limit_segments:
                    limits.append(limit)
                for est_speed in self.estimated_velocity_limit_segments:
                    estspeeds.append(est_speed)

                new_row = {'StartStationName': start_station[0], 'StartStationPosition': start_station[1], \
                           'EndStationName': end_station[0], 'EndStationPosition': end_station[1], 'MinJourneyTime': self.min_res_time, \
                           'PlannedJourneyTime': self.journey_time, 'JourneySpeedLimitSegments': limits, 'EstimatedSpeedPerSegment': estspeeds}

                self.df_journeys.loc[len(self.df_journeys)] = new_row

                i += 1

        excel_file_name = 'files/journeys.xlsx'
        self.df_journeys.to_excel(excel_file_name, index = False)

        print('INFO: Getting all possible journeys done and data saved to {}.'.format(excel_file_name))


    def select_journey(self, arbitrary=False, by_id=False, journey_id=None, from_xmljourney=False, start=None, stop=None, duration=None):               # angepasste Funktion

        if arbitrary or by_id:

            if by_id:
                assert isinstance(journey_id, int) and 0 <= journey_id <= len(self.df_journeys)-1, "Invalid Journey ID."

            journey_index = random.randint(0, len(self.df_journeys)-1-10) if arbitrary else journey_id                                                  # Anpassung, um 10 Eval-journeys auszunehmen

            self.journey_time = self.df_journeys.at[journey_index, 'PlannedJourneyTime']                                                                # vorerst belassen, da mit willkürlichem Faktor (hier 1.2) berechnet und auch bei Änderung der Art der Berechnung ähnlich willkürlich ermittelt werden würde

            self.journey.starting_point[0] = self.df_journeys.at[journey_index, 'StartStationPosition']
            self.journey.starting_point[1] = 0
            self.journey.starting_point[2] = self.df_journeys.at[journey_index, 'StartStationName']

            self.journey.stopping_point[0] = self.df_journeys.at[journey_index, 'EndStationPosition']
            self.journey.stopping_point[1] = np.round(self.journey_time / 60, 2)                                                                        # vorerst belassen, da mit willkürlichem Faktor (hier 1.2) berechnet und auch bei Änderung der Art der Berechnung ähnlich willkürlich ermittelt werden würde
            self.journey.stopping_point[2] = self.df_journeys.at[journey_index, 'EndStationName']

            self.journey_speed_limit_segments = ast.literal_eval(self.df_journeys.at[journey_index, 'JourneySpeedLimitSegments'])                       # Expertentfernung: vorerst noch im Code, da v_est auch für Diagramme verwendet wird und zur optischen Abschätzung der Ergebnisse beiträgt
            self.estimated_velocity_limit_segments = ast.literal_eval(self.df_journeys.at[journey_index, 'EstimatedSpeedPerSegment'])                   # Expertentfernung: vorerst noch im Code, da v_est auch für Diagramme verwendet wird und zur optischen Abschätzung der Ergebnisse beiträgt

        else:
            
            if from_xmljourney:

                self.journey.get_journey()
                self.journey_time = (self.journey.stopping_point[1] - self.journey.starting_point[1]) * 60  # in seconds
                self.minimal_time_distribution(self.journey.starting_point[0], 0.0)

            else:

                assert start in [i for i in range(0, len(self.stations.stations))], \
                    'Starting Point Error.'
                
                assert stop in [i for i in range(0, len(self.stations.stations))], \
                    'Stopping Point Error.'
                
                assert start < stop, 'Starting Point must come before the Stopping Point.'

                assert isinstance(duration, int), 'Please give duration in minutes (integer).'

                self.journey.starting_point[0] = self.stations.stations[start][1]
                self.journey.starting_point[1] = 0
                self.journey.starting_point[2] = self.stations.stations[start][0]

                self.journey.stopping_point[0] = self.stations.stations[stop][1]
                self.journey.stopping_point[1] = duration
                self.journey.stopping_point[2] = self.stations.stations[stop][0]

                self.journey_time = duration * 60
                self.minimal_time_distribution(self.journey.starting_point[0], 0.0)

        # self.mtd_plot()


    def _get_accumulated_time_from_estimates(self, position):                                                                                           # Expertentfernung: wird nurnoch für den randomstart verwendet

        elapsed_time = 0.0

        if self.train.position > self.journey.stopping_point[0]:
            elapsed_time = self.journey_time                                                                                                            # Anpassung: Übergang zu tatsächlichem Zeitfehler in step für den Endbereich (eigentlich irrelevant, da nicht mehr verwendet)
        
        elif position == self.journey.starting_point[0]:
            elapsed_time = 0.0
        
        else:
            for index, segment in enumerate(self.journey_speed_limit_segments):
                if segment[0] < position <= segment[1]:
                    elapsed_time += (position - segment[0]) / self.estimated_velocity_limit_segments[index]
                    break
                else:
                    elapsed_time += (segment[1] - segment[0]) / self.estimated_velocity_limit_segments[index]

        return elapsed_time


    def reset(self, reset_mode='arbitrary', journey_id=None, random_start=True):
        '''This method resets the environment to an initial state
        
        reset_mode: str
            'arbitrary': chooses a random journey from the excel file ./files/journeys.xlsx
            'from_xml': journey is read from the xml file ./files/journey.xml
            'by_id': choose a specific journey from the excel file ./files/journeys.xlsx. ID of journey must be provided

        journey_id: int
            must be provided if reset_mode='arbitrary'. It is the ID of the journey to use from the excel file ./files/journeys.xlsx

        random_start: bool
            True: journey starts at a random position between starting point and stopping point
                  train speed is also randomly selected with respect to estimated speed
            False: journey starts at starting point
            
        '''

        self.observation = (                                                                                                                            # angepasst und verschoben: reset der obs auf 0 erst am Anfang der reset Funktion, damit letzter step noch mit normaler obs gemacht wird
            np.array([0.0] * 10),
            np.zeros((27,2))
        )

        # select journey
        if reset_mode == 'arbitrary':
            self.select_journey(arbitrary=True)
        elif reset_mode == 'from_xml':
            self.select_journey(from_xmljourney=True)
        elif reset_mode == 'by_id':
            assert isinstance(journey_id, int), "Missing Journey ID" 
            self.select_journey(by_id=True, journey_id=journey_id)

        # reset train odometry data
        if random_start:

            random_value = random.random()

            if random_value < 0.5:
                # start from starting point
                self.train.position = self.journey.starting_point[0]
                self.train.speed = 0.0
                self.accumulated_journey_time = 0

            else:
                # start from 2000 metres before destination if journey distance > 3000 m
                if abs(self.journey.stopping_point[0] - self.journey.starting_point[0]) > 3000:
                    self.train.position = self.journey.stopping_point[0] - random.uniform(1000, 2000)
                    self.get_speed_limit_data(self.train.position)                                                                                      # Expertentfernung: ergänzt
                    #self.get_estimated_speed(self.train.position)                                                                                      # für Expertentfernung auskommentiert
                    #self.train.speed = self.v_est_current + random.uniform(0, 6)                                                                       # für Expertentfernung auskommentiert
                    self.train.speed = (self.speed_limit * 0.75) + random.uniform(0, 6)                                                                 # für Expertentfernung ergänzt/angepasst
                    self.accumulated_journey_time = self._get_accumulated_time_from_estimates(self.train.position)                                      # Expertentfernung: vorerst so belassen, da relativ willkürliche Zeit durch Faktor 1.2 und es sich nicht um einen direkten Eingriff in das Verhalten des agents handelt, also nicht um einen Expert
                else:
                    self.train.position = self.journey.starting_point[0]
                    self.train.speed = 0.0
                    self.accumulated_journey_time = 0

        else:
            self.train.position = self.journey.starting_point[0]
            self.train.speed = 0.0
            self.accumulated_journey_time = 0

        self.delta_time_left = self.accumulated_journey_time - self.journey_time                                                                        # für Expertentfernung angepasst um nicht mit 0 initialisiert zu werden, sondern dem tatsächlichen Zeitfehler

        self.initial_position = self.train.position                                                                                                     # Hier initialisiert, um die insgesamt zurückgelegte Strecke zu ermitteln anstatt der in dem step zurückgelegten Strecke (siehe step)

        # reset environment variables
        self.accumulated_action_time = 0

        # reset render variables
        self.render_speeds_arr = []
        self.render_positions_arr = []
        self.render_speeds_arr.append(self.train.speed)
        self.render_positions_arr.append(self.train.position)

        # reset distance to destination
        self.distance_to_destination = self.journey.stopping_point[0] - self.train.position                                                             # neu angepasst gegenüber mit Expert um den Bereich nach dem Endpunkt vom Bereich davor zu unterscheiden (hier eigentlich irrelevant)

        # get track data based on current position
        self.get_speed_limit_data(self.train.position)
        self.get_gradient_data(self.train.position)
        self.get_estimated_speed(self.train.position)                                                                                                   # Expertentfernung: vorerst noch im Code, da v_est auch für Diagramme verwendet wird und zur optischen Abschätzung der Ergebnisse beiträgt
        self.get_future_track_data()                                                                                                                    # Expertentfernung: Ermitteln der future_speed_limit_segments
    
        if not self.sequential_state:

            self.observation = (
                    0,                                                # current action
                    self.train.speed,                                 # current speed 
                    min(self.distance_to_destination, 100000),        # distance to next stopping point # experimentell: "2500" zu "100000", damit verbleibende Distanz "ungekürzt" an agent gegeben wird
                    self.v_est_current,                               # current estimated speed
                    min(self.delta_time_left, 3600),                  # actual time difference # experimentell: "300" zu "3600", damit Zeitfehler "ungekürzt" an agent gegeben wird
                    self.speed_limit,                                 # current speed limit
                    min(self.distance_next_speed_limit, 2500),        # current speed limit length
                    self.next_speed_limit,                            # next speed limit
                    self.current_gradient,                            # current gradient
                    min(self.distance_next_gradient, 2500),           # current gradient length
                    self.next_gradient                                # next gradient
            )

            return self._scale_state(np.array(self.observation, dtype=np.float32))
        
        else:                                                                                                                                           # Expertentfernung: angepasst auf neue Observation
            
            self.observation = (np.array([5.0, self.train.speed, min(self.distance_to_destination, 99990.0), 0.0, 
                                          min(self.delta_time_left, 3600), 0.0, 0.0, self.current_gradient, min(self.distance_next_gradient, 99990.0), self.next_gradient]), np.array(self.future_speed_limit_segments))

            return self._scale_state(self.observation)


    '''
    JOURNEY TIME DISTRIBUTION ALGORITHM
    '''

    def _minimal_speed_profile(self, current_position, current_speed):
        '''This method performs the steps 1,2, and 3 of the journey distribution algorithm'''
        
        '''
        PREPARATION:

        obtain the online data and the offline data including current speed,
        position and the speed limits during the journey.
        '''

        self.min_res_time = 0  # i.e using the minimal time speed profile
        self.journey_speed_limit_segments.clear()

        for segment in self.track.speed_limit_segments:

            if segment[0] <= current_position < segment[1]:
                if self.journey.stopping_point[0] > segment[1]:
                    self.journey_speed_limit_segments.append([current_position, segment[1], segment[2]])
                else:
                    self.journey_speed_limit_segments.append([current_position, self.journey.stopping_point[0], segment[2]])

            elif segment[0] >= current_position and segment[1] < self.journey.stopping_point[0]:
                self.journey_speed_limit_segments.append(segment)

            elif segment[0] <= self.journey.stopping_point[0] <= segment[1]:
                self.journey_speed_limit_segments.append([segment[0], self.journey.stopping_point[0], segment[2]])

        '''
        STEP 1:

        draw the maximum traction speed profiles from the train position and in
        each speed limit segment.
        '''

        self.minimal_time_profile_max_traction.clear()
        self.minimal_time_profile_max_traction.append([current_position, current_speed])

        for segment in self.journey_speed_limit_segments:

            while current_position <= segment[1]:
                
                if (current_position + self.delta_position) > self.journey.stopping_point[0]:
                    self.delta_position = self.journey.stopping_point[0] - current_position
                    current_acceleration = self.acceleration(self.max_traction, current_position, current_speed)
                
                    current_speed = math.sqrt(current_speed**2 + 2*current_acceleration*self.delta_position) \
                        if current_speed < segment[2] else segment[2]
                    current_position += self.delta_position

                    self.minimal_time_profile_max_traction.append([current_position, current_speed])

                    break
                
                current_acceleration = self.acceleration(self.max_traction, current_position, current_speed)

                current_speed = math.sqrt(current_speed**2 + 2*current_acceleration*self.delta_position) \
                    if current_speed < segment[2] else segment[2]
                current_position += self.delta_position

                self.minimal_time_profile_max_traction.append([current_position, current_speed])

        '''
        STEP 2:

        from the end of journey, draw the maximum braking speed 
        profile possible.
        '''

        self.minimal_time_profile_max_braking.clear()
        positions = list(reversed([point[0] for point in self.minimal_time_profile_max_traction]))
        self.minimal_time_profile_max_braking.append([positions[0], 0.0])
        current_position = positions[0]
        current_speed = 0.0

        iterator = 1
        
        for segment in reversed(self.journey_speed_limit_segments):

            while current_position >= segment[0]:

                if iterator > len(positions)-1:
                    break

                self.delta_position = current_position - positions[iterator]
                current_acceleration = self.acceleration(self.max_braking, current_position, current_speed)
                current_speed = math.sqrt(current_speed**2 + 2*abs(current_acceleration)*self.delta_position) \
                    if current_speed < segment[2] else segment[2]
                current_position = positions[iterator]
            
                self.minimal_time_profile_max_braking.append([current_position, current_speed])

                iterator +=1

        self.minimal_time_profile_max_braking.reverse()

        '''
        STEP 3:

        draw the minimal time speed profile
        '''

        self.minimal_time_profile.clear()
        for (tracting, braking) in zip(self.minimal_time_profile_max_traction, self.minimal_time_profile_max_braking):
            self.minimal_time_profile.append([tracting[0], min(tracting[1], braking[1])])


    def _minimal_time_distribution(self):
        '''This method perorms step 4 of journey time distribution algorithm'''

        '''
        STEP 4:

        calculate the minimum reserved time and the minimum reserved time per speed limit segment
        '''

        min_res_time_segments = []
        i = 0
        position = self.journey_speed_limit_segments[0][0]
        time_temp = 0.0
        
        for index, point in enumerate(self.minimal_time_profile):

            if index < len(self.minimal_time_profile) - 1:

                s = abs(point[0] - self.minimal_time_profile[index+1][0])
                position += s
                u = point[1]
                v = self.minimal_time_profile[index+1][1]
                a = (v**2 - u**2) / (2*s) if s != 0 else (v**2 - u**2) / 0.001
                delta_t = (v - u) / a if a != 0 else s/v
                self.min_res_time += delta_t

                if position >= self.journey_speed_limit_segments[i][1]:
                    min_res_time_segments.append(self.min_res_time - time_temp)
                    time_temp = self.min_res_time
                    i += 1

                if position >= self.minimal_time_profile[-1][0]:
                    break

        return min_res_time_segments
    

    def _fitted_time_distribution(self, min_res_time_segments):
        '''This method performs last step of the journey time distribution algorithm'''

        '''
        STEP 5:
        
        Calculate reserved time in the current and next speed limit segments
        '''

        self.res_time_segments.clear()
        self.estimated_velocity_limit_segments.clear()
        difference = self.journey_time - self.accumulated_journey_time
        cummulative_min_reserved_time = 0

        for index, segment in enumerate(self.journey_speed_limit_segments):

            self.res_time_segments.append(difference * (min_res_time_segments[index] / (self.min_res_time - cummulative_min_reserved_time)))
            self.estimated_velocity_limit_segments.append((segment[1] - segment[0])/self.res_time_segments[-1])
            difference -= self.res_time_segments[index]
            cummulative_min_reserved_time += min_res_time_segments[index]

        self.estimated_velocity_limit_segments.append(0.0)


    def minimal_time_distribution(self, current_position, current_speed):
        '''This method performs all steps of the journey time distribution algorithm'''

        self._minimal_speed_profile(current_position, current_speed)
        min_res_time_per_seg = self._minimal_time_distribution()
        self._fitted_time_distribution(min_res_time_per_seg)


    def mtd_plot(self):
        '''Plot resultant plots of the journey time distribution algorithm'''

        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2)
        fig.set_figwidth(10)
        fig.set_figheight(2*3)

        # Plot maximum traction profile
        axs[0,0].step([segment[0]/1000 for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][1]/1000],
                    [segment[2] for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][2]],
                    where='post', color='red', label='Speed Limit')
        axs[0,0].plot([point[0]/1000 for point in self.minimal_time_profile_max_traction], [point[1] for point in self.minimal_time_profile_max_traction], color='olive', label='Maximum Traction Profile')
        axs[0,0].set_ylim([0.0, 50.0])
        axs[0,0].set_xlim([3.419, 22.197])
        axs[0,0].grid(color='lightgrey', linestyle='--')
        axs[0,0].set_xlabel('Position [m]')
        axs[0,0].set_ylabel('Speed [m/s]')
        axs[0,0].set_title('Step 1: Maximum Traction Profile')
        axs[0,0].legend()

        # Plot maximum braking profile
        axs[0,1].step([segment[0]/1000 for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][1]/1000],
                    [segment[2] for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][2]],
                    where='post', color='red', label='Speed Limit')
        axs[0,1].plot([point[0]/1000 for point in self.minimal_time_profile_max_braking], [point[1] for point in self.minimal_time_profile_max_braking], color='gold', label='Maximum Braking Profile')
        axs[0,1].set_ylim([0.0, 50.0])
        axs[0,1].set_xlim([3.419, 22.197])
        axs[0,1].grid(color='lightgrey', linestyle='--')
        axs[0,1].set_xlabel('Position [m]')
        axs[0,1].set_ylabel('Speed [m/s]')
        axs[0,1].set_title('Step 2: Maximum Braking Profile')
        axs[0,1].legend()

        # Plot minimum time profile
        axs[1,0].step([segment[0]/1000 for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][1]/1000],
                    [segment[2] for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][2]],
                    where='post', color='red', label='Speed Limit')
        axs[1,0].plot([point[0]/1000 for point in self.minimal_time_profile], [point[1] for point in self.minimal_time_profile], color='blue', label='Minimal Time Profile')
        axs[1,0].set_ylim([0.0, 50.0])
        axs[1,0].set_xlim([3.419, 22.197])
        axs[1,0].grid(color='lightgrey', linestyle='--')
        axs[1,0].set_xlabel('Position [m]')
        axs[1,0].set_ylabel('Speed [m/s]')
        axs[1,0].set_title('Step 3: Minimum Time Profile')
        axs[1,0].legend()

        # Plot average speed profile
        axs[1,1].step([segment[0]/1000 for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][1]/1000],
                    [segment[2] for segment in self.journey_speed_limit_segments] + [self.journey_speed_limit_segments[-1][2]],
                    where='post', color='red', label='Speed Limit')
        for index, segment in enumerate(self.journey_speed_limit_segments):
            if index == 0:
                axs[1,1].plot([segment[0]/1000, segment[1]/1000], [self.estimated_velocity_limit_segments[index], self.estimated_velocity_limit_segments[index]], color='green', label='Estimated Speed')
            else:
                axs[1,1].plot([segment[0]/1000, segment[1]/1000], [self.estimated_velocity_limit_segments[index], self.estimated_velocity_limit_segments[index]], color='green')
        axs[1,1].set_ylim([0.0, 50.0])
        axs[1,1].set_xlim([3.419, 22.197])
        axs[1,1].grid(color='lightgrey', linestyle='--')
        axs[1,1].set_xlabel('Position [m]')
        axs[1,1].set_ylabel('Speed [m/s]')
        axs[1,1].set_title('Step 4-6: Average Speed Profile')
        axs[1,1].legend()
        
        plt.legend()
        plt.show()


    '''
    CHOOSE ACTION
    '''

    def choose_action(self, expert=False):

        '''Choose Action
        
        This method is used to choose a random action incase expert isn't activated.
        Incase expert is activated, the choice of actions can be influenced.
        '''

        if not expert:
            # Choose random action
            return random.choice([0,1,2,3,4,5,6,7,8,9,10])

        else:
            # Influence choice of action
            self.get_estimated_speed(self.train.position)
            distance_to_destination = self.journey.stopping_point[0] - self.train.position

            critical_zone = False
            min_braking_distance = 0
            safety_distance = 200

            if self.train.speed < self.next_speed_limit and self.v_est_next != 0.0:
                critical_zone = False

            else:
                current_position = self.train.position
                current_speed = self.train.speed

                target_speed = self.next_speed_limit
                if self.v_est_next == 0.0:
                    target_speed = 0.0

                while current_speed >= target_speed:

                    deceleration = self.acceleration(action=-100, current_position=current_position, current_speed=current_speed)
                    delta_position = current_speed * self.delta_t + 0.5 * deceleration * (self.delta_t ** 2)
                    min_braking_distance += delta_position
                    current_position += delta_position
                    current_speed = current_speed + deceleration * self.delta_t

                if self.distance_next_speed_limit > min_braking_distance + safety_distance and \
                    distance_to_destination > min_braking_distance + np.random.randint(0, 120):
                    critical_zone = False
                else:
                    critical_zone = True
            
            if not critical_zone:

                if self.speed_limit + 5 > self.train.speed > self.v_est_current - 10: 
                    return random.choice([-5,-3,-1,0,1,3,5])
                elif self.train.speed >= self.speed_limit + 5:
                    return random.choice([-5,-3,-1,0])
                else:
                    return random.choice([1,3,5])

            else:
                if random.random() > 0.5:
                    return random.choice([-5,-3,-1,0])
                else:
                    return random.choice([-5,-3,-1,0,1,3,5])


    '''
    RENDERING
    '''

    def _hs_trf(self, ys): 
        '''Transform world speed to scaled visualization'''
        return ys * self.scale_height_speed + 50


    def _hg_trf(self, yg): 
        '''Transform world slope to scaled visualization'''
        return (yg + 30) * self.scale_height_slope + 50


    def _w_trf(self, x):
        '''transform world position to scaled visualization'''
        return (x - self.journey.starting_point[0]) * self.scale_width + 50


    def render(self, mode='human', close=False):
        '''Display state of train dependent on the actions taken'''

        self.rendering = True # Flag to let other modules know rendering is active

        if self.render_mode is None:

            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:

            import pygame
            from pygame import gfxdraw

        except ImportError:

            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:

            pygame.init()

            if self.render_mode == 'human':

                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption('Train Environment')

            else:

                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        if self.clock is None:

            self.clock = pygame.time.Clock()

        self.surface = pygame.Surface((self.screen_width, self.screen_height))
        self.surface.fill(self.white)

        self.world_width = self.journey.stopping_point[0] - self.journey.starting_point[0]
        self.scale_width = (self.screen_width - 100) / self.world_width
    
        text_format = pygame.font.SysFont('consola.ttf', 16)
        text_format_title = pygame.font.SysFont('consola.ttf', 32)

        # Visualization Title
        txt_surf = text_format_title.render("Train Environment", True, self.black, self.white)
        txt_rect = txt_surf.get_rect()
        txt_surf = pygame.transform.flip(txt_surf, False, True)
        txt_rect.center = (500, 375)
        self.surface.blit(txt_surf, txt_rect)

        # draw gradient profile
        world_points = []
        check = False

        for segment in self.track.gradient_segments:

            if segment[0] < self.journey.starting_point[0] <= segment[1]:

                check = True
                world_points.append((self.journey.starting_point[0], segment[2]))

                if segment[0] < self.journey.stopping_point[0] <= segment[2]:

                    world_points.append((self.journey.stopping_point[0], segment[2]))
                    check = False
                    break

                else:

                    world_points.append((segment[1], segment[2]))

            elif check is True:

                if not (segment[0] < self.journey.stopping_point[0] <= segment[1]):

                    world_points.append((segment[0], segment[2]))
                    world_points.append((segment[1], segment[2]))

                else:

                    world_points.append((segment[0], segment[2]))
                    world_points.append((self.journey.stopping_point[0], segment[2]))
                    check = False
                    break

        scaled_points_slope = [(self._w_trf(pt[0]), self._hg_trf(pt[1])) for pt in world_points]
        scaled_points_slope.extend(((950,50),(50,50)))
        gfxdraw.filled_polygon(self.surface, scaled_points_slope, self.lightgrey)

        # plot speed axes
        speed = 0
        while speed <= int(np.floor(self.max_speed_limit/ 10) * 10):

            # pygame.draw.line(self.surface, self.lightblue, (50, self._hs_trf(speed)), (950, self._hs_trf(speed)))
            txt_surf = text_format.render(str(speed) + " m/s", True, self.grey, self.white)
            txt_rect = txt_surf.get_rect()
            txt_surf = pygame.transform.flip(txt_surf, False, True)
            txt_rect.midleft = (5, self._hs_trf(speed))
            self.surface.blit(txt_surf, txt_rect)
            speed += 10

        # plot gradient axes
        for gradient in range(-20, 21, 10):

            if gradient == 0:

                pygame.draw.line(self.surface, self.lightlightgrey, (50, self._hg_trf(gradient)), (950, self._hg_trf(gradient)))

            txt_surf = text_format.render(str(gradient) + " %", True, self.grey, self.white)
            txt_rect = txt_surf.get_rect()
            txt_surf = pygame.transform.flip(txt_surf, False, True)
            txt_rect.midleft = (963, self._hg_trf(gradient))
            self.surface.blit(txt_surf, txt_rect)

        # plot speed limits
        world_points = []
        check = False

        for segment in self.track.speed_limit_segments:

            if segment[0] < self.journey.starting_point[0] <= segment[1]:

                check = True
                world_points.append((self.journey.starting_point[0], segment[2]))

                if segment[0] < self.journey.stopping_point[0] <= segment[2]:

                    world_points.append((self.journey.stopping_point[0], segment[2]))
                    check = False
                    break

                else:

                    world_points.append((segment[1], segment[2]))

            elif check is True:

                if not (segment[0] < self.journey.stopping_point[0] <= segment[1]):

                    world_points.append((segment[0], segment[2]))
                    world_points.append((segment[1], segment[2]))

                else:

                    world_points.append((segment[0], segment[2]))
                    world_points.append((self.journey.stopping_point[0], segment[2]))
                    check = False
                    break
        
        scaled_points_speed_limit = [(self._w_trf(pt[0]), self._hs_trf(pt[1])) for pt in world_points]
        pygame.draw.lines(self.surface, self.red, False, scaled_points_speed_limit, 2)

        # plot bounding box
        bounding_box_coords = pygame.Rect(50, 50, self.screen_width - 100, self.screen_height - 100)
        pygame.draw.rect(self.surface, self.black, bounding_box_coords, 1)

        # show starting and stopping points
        pygame.draw.polygon(self.surface, self.black, [(self._w_trf(self.journey.starting_point[0])-5,70),\
            (self._w_trf(self.journey.starting_point[0]),50),(self._w_trf(self.journey.starting_point[0])+5,70)])
            
        pygame.draw.polygon(self.surface, self.black, [(self._w_trf(self.journey.stopping_point[0])-5,70),\
            (self._w_trf(self.journey.stopping_point[0]),50),(self._w_trf(self.journey.stopping_point[0])+5,70)])

        txt_surf = text_format.render("Start", True, self.black, self.white)
        txt_rect = txt_surf.get_rect()
        txt_surf = pygame.transform.flip(txt_surf, False, True)
        txt_rect.center = (self._w_trf(self.journey.starting_point[0]), 80)
        self.surface.blit(txt_surf, txt_rect)

        txt_surf = text_format.render(str(self.journey.starting_point[0])+" m", True, self.black, self.white)
        txt_rect = txt_surf.get_rect()
        txt_surf = pygame.transform.flip(txt_surf, False, True)
        txt_rect.center = (self._w_trf(self.journey.starting_point[0]), 40)
        self.surface.blit(txt_surf, txt_rect)

        txt_surf = text_format.render("Stop", True, self.black, self.white)
        txt_rect = txt_surf.get_rect()
        txt_surf = pygame.transform.flip(txt_surf, False, True)
        txt_rect.center = (self._w_trf(self.journey.stopping_point[0]), 80)
        self.surface.blit(txt_surf, txt_rect)

        txt_surf = text_format.render(str(self.journey.stopping_point[0])+" m", True, self.black, self.white)
        txt_rect = txt_surf.get_rect()
        txt_surf = pygame.transform.flip(txt_surf, False, True)
        txt_rect.center = (self._w_trf(self.journey.stopping_point[0]), 40)
        self.surface.blit(txt_surf, txt_rect)

        # plot speed points as train moves along track
        world_speed_points = [(self.render_positions_arr[i], self.render_speeds_arr[i]) for i in range(len(self.render_positions_arr))]
        scaled_speed_points = [(self._w_trf(pt[0]), self._hs_trf(pt[1])) for pt in world_speed_points]
        scaled_speed_points.insert(0,(50, 50))
        pygame.draw.lines(self.surface, self.blue, False, scaled_speed_points)

        # draw position (and maybe speed) pointing line
        pos = self._w_trf(self.train.position)
        speed = self._hs_trf(self.train.speed)
        speed_limit = self._hs_trf(self.speed_limit)
        current_gradient = self._hg_trf(self.render_gradient)

        pygame.draw.line(self.surface, self.blue, (pos, 50), \
            (pos, self.screen_height - 50))

        pygame.draw.polygon(self.surface, self.blue, [(pos-10,70),(pos,50),(pos+10,70)])

        # draw speed, speed limit and gradient markers
        pygame.draw.polygon(self.surface, self.blue, [(50,speed),(40,speed-5),(40,speed+5)]) # speed
        pygame.draw.polygon(self.surface, self.red, [(50,speed_limit),(40,speed_limit-5),(40,speed_limit+5)]) # speed_limit
        pygame.draw.polygon(self.surface, self.lightgrey, [(950,current_gradient),(960,current_gradient-5),(960,current_gradient+5)]) # gradient

        # write current position
        position_surf = text_format.render(str(np.round(self.train.position, 1)) + " m", True, self.blue, self.white)
        position_rect = position_surf.get_rect()
        position_surf = pygame.transform.flip(position_surf, False, True)
        position_rect.center = (self._w_trf(self.train.position), 40)
        self.surface.blit(position_surf, position_rect)

        # write current speed
        speed_txt_surf = text_format.render(str(np.round(self.train.speed, 2)) + " m/s", True, self.blue, self.white)
        speed_txt_rect = speed_txt_surf.get_rect()
        speed_txt_surf = pygame.transform.flip(speed_txt_surf, False, True)
        speed_txt_rect.center = (self._w_trf(self.train.position), 25)
        self.surface.blit(speed_txt_surf, speed_txt_rect)

        # write current acceleration
        acceleration_surf = text_format.render(str(np.round(self.train.acceleration, 2)) + " m/s^2", True, self.blue, self.white)
        acceleration_rect = acceleration_surf.get_rect()
        acceleration_surf = pygame.transform.flip(acceleration_surf, False, True)
        acceleration_rect.center = (self._w_trf(self.train.position), 10)
        self.surface.blit(acceleration_surf, acceleration_rect)

        self.surface = pygame.transform.flip(self.surface, False, True)
        self.screen.blit(self.surface, (0,0))

        if self.render_mode == 'human':

            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == 'rgb_array':

            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1,0,2)
            )

        # handle events
        for event in pygame.event.get():

            # if x button at top right of pygame window is pressed
            if event.type == pygame.QUIT:

                pygame.quit()
                exit()
     

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.rendering = False