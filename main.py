import torch
import gym
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from progressbar import progressbar
import argparse
from pathlib import Path
from gym_train.envs import TrainEnv
from agentDDQN import DDQNAgent
from agentPPO import PPOAgent
from utils.model import CustomActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from progressbar import progressbar
from tqdm import tqdm                                                                                                                                   # Bibliothek für Fortschrittsbalken
from stable_baselines3 import DQN                                                                                                                       # DQN Algorithmus
from stable_baselines3 import A2C                                                                                                                       # A2C Algorithmus
from sb3_contrib import RecurrentPPO                                                                                                                    # RecurrentPPO Algorithmus
from sb3_contrib import QRDQN                                                                                                                           # QRDQN Algorithmus
from sb3_contrib import TRPO                                                                                                                            # TRPO Algorithmus
import optuna                                                                                                                                           # optuna: Hyperparametertuning

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)      
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", str(device))                                                                                                                           # print des verwendeten devices

parser = argparse.ArgumentParser(description='DRL Train Speed Optimizer')

parser.add_argument('-a', '--algo', type=str, choices=['PPO', 'DQN', 'A2C', 'RecurrentPPO', 'QRDQN', 'TRPO'], help='Reinforcement learning algorithm to used.')                                         # parser: erweitert für neue Algorithmen
parser.add_argument('-t', '--train', action='store_const', const=True, help='Activate agent training mode.')
parser.add_argument('-tp', '--training_path', type=str, help='Path to location where training data and logging data will be saved.')
parser.add_argument('-ep', '--evaluation_path', type=str, help='Path to location with agent that should be evaluated.')
parser.add_argument('-an', '--agent_name', type=str, help="Name that the agent is saved as in the evaluation folder. Please end name with '.zip'")
parser.add_argument('-jm', '--journey_mode', type=int, choices=[0, 1, 2], help='0: Random Journey, 1: Journey from xml, 2: Journey by ID')
parser.add_argument('-id', '--journey_id', type=int, help='ID of journey along which agent shouzld be evaluated.')
parser.add_argument('-r', '--render', action='store_const', const=True, help='Render while evaluating performance.')
parser.add_argument('-uj', '--update_journeys', action='store_const', const=True, help='Recalculate journey data using journey time distribution algorithm and save to excel file.')
parser.add_argument('-ht', '--hyperparameter_tuning', action='store_const', const=True, help='Activate hyperparameter tuning mode.')                                                                    # für optuna: Hyperparameter-tuning-modus
parser.add_argument('-htp', '--hyperparameter_tuning_path', type=str, help='Path to location where hyperparameter-tuning data will be saved.')                                                          # für optuna: Pfad für das Hyperparameter-tuning

args = parser.parse_args()


optuna_iter_counter = int(0)                                                                                                                            # Zähler für die optuna-Iterationen 

hyperparameter_folder = Path(args.hyperparameter_tuning_path) if args.hyperparameter_tuning == True else None                                           # für optuna: Pfad für das Hyperparameter-tuning wird in Variable geschrieben

if args.update_journeys:
    # Perform the Journey Time Distribution Algorithm for all possible journeys and save their corresponding data to an Excel table

    print('INFO: Updating all possible journey data for training ...')

    env_ = gym.make('gym_train:train-v0')
    env_.get_all_possible_journeys()

    print('INFO: Updating done!')

# create parallel environments for PPO training with Stable Baselines 3
env = make_vec_env('gym_train:train-v0', n_envs=4)                                                                                                      # Angepasst, um für alle Algorithmen ein vec_env zu erstellen

def optimize_agent(trial):                                                                                                                              # Funktion für optuna

    global optuna_iter_counter                                                                                                                          # optuna_iter_counter für diese Funktion zugänglich machen

    if args.algo == 'PPO': # PPO learning algorithm                                                                                                     # PPO: Festlegen der Wertebereiche und Schrittgrößen der Hyperparameter für optuna und Erstellen des models

        learning_rate       = trial.suggest_float('learning_rate', 0.00001, 0.01, step=0.00001)    # SB3 Standard: 0.0003
        n_steps             = trial.suggest_int('n_steps', 128, 4096, step=32)    # SB3 Standard: 2048
        batch_size          = trial.suggest_int('batch_size', 16, 512, step=16)    # SB3 Standard: 64
        n_epochs            = trial.suggest_int('n_epochs', 1, 20)    # SB3 Standard: 10
        gamma               = trial.suggest_float('gamma', 0.8, 0.999, step=0.001)    # SB3 Standard: 0.99
        gae_lambda          = trial.suggest_float('gae_lambda', 0.9, 1.0, step=0.001)    # SB3 Standard: 0.95
        clip_range          = trial.suggest_float('clip_range', 0.1, 0.3, step=0.001)    # SB3 Standard: 0.2
        ent_coef            = trial.suggest_float('ent_coef', 0.0, 0.05, step=0.001)    # SB3 Standard: 0.0
        vf_coef             = trial.suggest_float('vf_coef', 0.5, 1.0, step=0.001)    # SB3 Standard: 0.5
        max_grad_norm       = trial.suggest_float('max_grad_norm', 0.1, 10.0, step=0.01)    # neu, SB3 Standard: 0.5
        ortho_init          = False
        activation_fn       = torch.nn.ReLU
        net_arch            = dict(pi=[256,256], vf=[256,256])
        normalize_images    = False
        policy_kwargs       = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, normalize_images=normalize_images)
        time_steps          = trial.suggest_int('time_steps', 5000, 1600000, step=5000)

        model = PPO(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, device=device, verbose=0, 
            tensorboard_log=hyperparameter_folder, policy_kwargs=policy_kwargs, seed=42)
        
    elif args.algo == 'DQN': # DQN learning algorithm                                                                                                   # DQN: Festlegen der Wertebereiche und Schrittgrößen der Hyperparameter für optuna und Erstellen des models
        
        learning_rate       = trial.suggest_float('learning_rate', 0.00001, 0.01, step=0.00001)    # SB3 Standard: 0.0001
        buffer_size         = trial.suggest_int('buffer_size', 10000, 10000000, step=10000)    # SB3 Standard: 1000000
        learning_starts     = trial.suggest_int('learning_starts', 10, 10000, step=10)    # SB3 Standard: 100
        batch_size          = trial.suggest_int('batch_size', 16, 512, step=16)    # SB3 Standard: 32
        tau                 = trial.suggest_categorical('tau', [0.005, 0.1, 0.5, 0.95, 1.0])    # SB3 Standard: 1.0
        gamma               = trial.suggest_float('gamma', 0.8, 0.999, step=0.001)    # SB3 Standard: 0.99
        train_freq          = trial.suggest_int('train_freq', 1, 8)    # SB3 Standard: 4
        gradient_steps      = trial.suggest_int('gradient_steps', 1, 10)    # neu, SB3 Standard: 1
        target_update_interval = trial.suggest_int('target_update_interval', 500, 100000, step=500)    # SB3 Standard: 10000
        exploration_fraction = trial.suggest_float('exploration_fraction', 0.01, 0.9, step=0.01)    # SB3 Standard: 0.1
        exploration_initial_eps = trial.suggest_categorical('exploration_initial_eps', [0.5, 0.75, 1.0])    # SB3 Standard: 1.0
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.01, 0.1, step=0.001)    # SB3 Standard: 0.05
        max_grad_norm       = trial.suggest_float('max_grad_norm', 0.1, 20.0, step=0.01)    # neu, SB3 Standard: 10
        activation_fn       = torch.nn.ReLU
        net_arch            = [256, 256]
        normalize_images    = False
        policy_kwargs       = dict(activation_fn=activation_fn, net_arch=net_arch, normalize_images=normalize_images)
        time_steps          = trial.suggest_int('time_steps', 5000, 1600000, step=5000)

        model = DQN(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, 
            batch_size=batch_size, tau=tau, gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval, 
            exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, max_grad_norm=max_grad_norm, tensorboard_log=hyperparameter_folder, 
            policy_kwargs=policy_kwargs, verbose=0, device=device, seed=42)
        
    elif args.algo == 'A2C': # A2C learning algorithm                                                                                                   # A2C: Festlegen der Wertebereiche und Schrittgrößen der Hyperparameter für optuna und Erstellen des models

        learning_rate       = trial.suggest_float('learning_rate', 0.00001, 0.01, step=0.00001)    # SB3 Standard: 0.0007
        n_steps             = trial.suggest_int('n_steps', 1, 50)    # SB3 Standard: 5
        gamma               = trial.suggest_float('gamma', 0.8, 0.999, step=0.001)    # SB3 Standard: 0.99
        gae_lambda          = trial.suggest_float('gae_lambda', 0.9, 1.0, step=0.001)    # SB3 Standard: 1.0
        ent_coef            = trial.suggest_float('ent_coef', 0.0, 0.05, step=0.001)    # SB3 Standard: 0.0
        vf_coef             = trial.suggest_float('vf_coef', 0.5, 1.0, step=0.001)    # SB3 Standard: 0.5
        max_grad_norm       = trial.suggest_float('max_grad_norm', 0.1, 10.0, step=0.01)    # SB3 Standard: 0.5
        rms_prop_eps        = trial.suggest_float('rms_prop_eps', 0.000001, 0.001, step=0.000001)    # SB3 Standard: 0.00001
        use_rms_prop        = trial.suggest_categorical('use_rms_prop', [True, False])    # SB3 Standard: True
        ortho_init          = False
        activation_fn       = torch.nn.ReLU
        net_arch            = dict(pi=[256,256], vf=[256,256])
        normalize_images    = False
        policy_kwargs       = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, normalize_images=normalize_images)
        time_steps          = trial.suggest_int('time_steps', 5000, 1600000, step=5000)

        model = A2C(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps,
            gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps, use_rms_prop=use_rms_prop, device=device, verbose=0, 
            tensorboard_log=hyperparameter_folder, policy_kwargs=policy_kwargs, seed=42)
        
    elif args.algo == 'RecurrentPPO': # RecurrentPPO learning algorithm                                                                                 # RecurrentPPO: Festlegen der Wertebereiche und Schrittgrößen der Hyperparameter für optuna und Erstellen des models

        learning_rate       = trial.suggest_float('learning_rate', 0.00001, 0.01, step=0.00001)    # SB3 Standard: 0.0003
        n_steps             = trial.suggest_int('n_steps', 32, 4096, step=32)    # SB3 Standard: 128
        batch_size          = trial.suggest_int('batch_size', 16, 512, step=16)    # SB3 Standard: 128
        n_epochs            = trial.suggest_int('n_epochs', 1, 20)    # SB3 Standard: 10
        gamma               = trial.suggest_float('gamma', 0.8, 0.999, step=0.001)    # SB3 Standard: 0.99
        gae_lambda          = trial.suggest_float('gae_lambda', 0.9, 1.0, step=0.001)    # SB3 Standard: 0.95
        clip_range          = trial.suggest_float('clip_range', 0.1, 0.3, step=0.001)    # SB3 Standard: 0.2
        ent_coef            = trial.suggest_float('ent_coef', 0.0, 0.05, step=0.001)    # SB3 Standard: 0.0
        vf_coef             = trial.suggest_float('vf_coef', 0.5, 1.0, step=0.001)    # SB3 Standard: 0.5
        max_grad_norm       = trial.suggest_float('max_grad_norm', 0.1, 10.0, step=0.01)    # SB3 Standard: 0.5
        ortho_init          = False
        activation_fn       = torch.nn.ReLU
        net_arch            = dict(pi=[256,256], vf=[256,256])
        lstm_hidden_size    = trial.suggest_int('lstm_hidden_size', 16, 512, step=16)    # SB3 Standard: 256
        n_lstm_layers       = trial.suggest_int('n_lstm_layers', 1, 4, step=1)    # SB3 Standard: 1
        shared_lstm         = False    # SB3 Standard: False
        enable_critic_lstm  = True    # SB3 Standard: True
        normalize_images    = False
        policy_kwargs = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, lstm_hidden_size=lstm_hidden_size, n_lstm_layers=n_lstm_layers, shared_lstm=shared_lstm, enable_critic_lstm=enable_critic_lstm, normalize_images=normalize_images)
        time_steps          = trial.suggest_int('time_steps', 5000, 1600000, step=5000)

        model = RecurrentPPO(policy="MultiInputLstmPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, device=device, verbose=0, 
            tensorboard_log=hyperparameter_folder, policy_kwargs=policy_kwargs, seed=42)
        
    elif args.algo == 'QRDQN': # QR-DQN learning algorithm                                                                                              # QRDQN: Festlegen der Wertebereiche und Schrittgrößen der Hyperparameter für optuna und Erstellen des models

        learning_rate       = trial.suggest_float('learning_rate', 0.00001, 0.00015, step=0.00001)    # SB3 Standard: 5e-05
        buffer_size         = trial.suggest_int('buffer_size', 4000000, 7000000, step=10000)    # SB3 Standard: 1000000
        learning_starts     = trial.suggest_int('learning_starts', 3000, 5000, step=100)    # SB3 Standard: 100
        batch_size          = trial.suggest_int('batch_size', 128, 256, step=16)    # SB3 Standard: 32
        tau                 = trial.suggest_categorical('tau', [0.005, 0.1, 0.5, 0.95, 1.0])    # SB3 Standard: 1.0
        gamma               = trial.suggest_float('gamma', 0.9, 0.99, step=0.001)    # SB3 Standard: 0.99
        train_freq          = trial.suggest_int('train_freq', 1, 8)    # SB3 Standard: 4
        gradient_steps      = trial.suggest_int('gradient_steps', 1, 6)    # SB3 Standard: 1
        target_update_interval = trial.suggest_int('target_update_interval', 3000, 4300, step=100)    # SB3 Standard: 10000
        exploration_fraction = trial.suggest_float('exploration_fraction', 0.75, 0.99, step=0.01)    # SB3 Standard: 0.005
        exploration_initial_eps = trial.suggest_categorical('exploration_initial_eps', [0.75, 0.9, 1.0])    # SB3 Standard: 1.0
        exploration_final_eps = trial.suggest_float('exploration_final_eps', 0.05, 0.1, step=0.001)    # SB3 Standard: 0.01
        max_grad_norm       = trial.suggest_float('max_grad_norm', 0.0, 3.0, step=0.1)    # SB3 Standard: None
        if max_grad_norm == 0.0:
            max_grad_norm = None
        n_quantiles         = trial.suggest_int('n_quantiles', 100, 300, step=10)    # SB3 Standard: 200
        activation_fn       = torch.nn.ReLU
        net_arch            = [256, 256]
        normalize_images    = False
        policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch, n_quantiles=n_quantiles, normalize_images=normalize_images)
        time_steps          = trial.suggest_int('time_steps', 700000, 1000000, step=10000)

        model = QRDQN(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts,
            batch_size=batch_size, tau=tau, gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, max_grad_norm=max_grad_norm, 
            device=device, verbose=0, tensorboard_log=hyperparameter_folder, policy_kwargs=policy_kwargs, seed=42)

        
    elif args.algo == 'TRPO': # TRPO learning algorithm                                                                                                 # TRPO: Festlegen der Wertebereiche und Schrittgrößen der Hyperparameter für optuna und Erstellen des models

        learning_rate       = trial.suggest_float('learning_rate', 0.00001, 0.01, step=0.00001)    # SB3 Standard: 0.001
        n_steps             = trial.suggest_int('n_steps', 128, 4096, step=32)    # SB3 Standard: 2048
        batch_size          = trial.suggest_int('batch_size', 16, 512, step=16)    # SB3 Standard: 128
        gamma               = trial.suggest_float('gamma', 0.8, 0.999, step=0.001)    # SB3 Standard: 0.99
        cg_max_steps        = trial.suggest_int('cg_max_steps', 10, 30, step=1)    # SB3 Standard: 15
        cg_damping          = trial.suggest_float('cg_damping', 0.01, 0.2, step=0.01)    # SB3 Standard: 0.1
        line_search_shrinking_factor = trial.suggest_float('line_search_shrinking_factor', 0.5, 0.9, step=0.01)    # SB3 Standard: 0.8
        line_search_max_iter = trial.suggest_int('line_search_max_iter', 1, 50, step=1)    # SB3 Standard: 10
        n_critic_updates    = trial.suggest_int('n_critic_updates', 1, 20, step=1)    # SB3 Standard: 10
        gae_lambda          = trial.suggest_float('gae_lambda', 0.9, 1.0, step=0.001)    # SB3 Standard: 0.95
        sub_sampling_factor = trial.suggest_float('sub_sampling_factor', 0.1, 1, step=0.01)    # SB3 Standard: 1
        ortho_init          = False
        activation_fn       = torch.nn.ReLU
        net_arch            = dict(pi=[256,256], vf=[256,256])
        normalize_images    = False
        policy_kwargs = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, normalize_images=normalize_images)
        time_steps          = trial.suggest_int('time_steps', 5000, 1600000, step=5000)

        model = TRPO(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, gamma=gamma,
            cg_max_steps=cg_max_steps, cg_damping=cg_damping, line_search_shrinking_factor=line_search_shrinking_factor, 
            line_search_max_iter=line_search_max_iter, n_critic_updates=n_critic_updates, gae_lambda=gae_lambda, sub_sampling_factor=sub_sampling_factor, device=device, 
            verbose=0, tensorboard_log=hyperparameter_folder, policy_kwargs=policy_kwargs, seed=42)

    num_iters = trial.suggest_int('num_iters', 1, 5)                                                                                                    # Anzahl der Iterationen als zusätzlichen tuning-Parameter mit Wertebereich für optuna
    for iter in range(num_iters):                                                                                                                       # Trainingsschleife des models (Lernen, Tensorboard-logging und Speichern der angelernten models)
        model.learn(total_timesteps=time_steps, tb_log_name=f'{str(optuna_iter_counter)+args.algo}_tb_log'+str(iter), reset_num_timesteps=False)
        model.save(f"{hyperparameter_folder}/{str(optuna_iter_counter)+args.algo+'agent'+str(iter)}")

    optuna_iter_counter += 1

    eval_env = gym.make('gym_train:train-v0')                                                                                                           # Erstellen des environments für die Evaluierungen

    rewards_counter = 0                                                                                                                                 # Variablen für die an Optuna übergebenen Evaluierungsergebnisse initialisieren
    parking_error_counter = 0
    punctuality_error_counter = 0

    for journey_eval_counter in range(10):                                                                                                              # Evaluierungsschleife des angelernten models (über die 10 letzten journeys aus dem journeys-file)

        journey_eval_id = (journey_eval_counter + 110)                                                                                                  # Auswahl der journey-id (startend bei 110 bis 119 für die 10 letzten journeys aus dem journeys-file)

        rewards = 0                                                                                                                                     # Variablen für die Evaluierungsergebnisse der einzelnen journeys initialisieren
        parking_error = 0
        punctuality_error = 0

        if args.algo != 'RecurrentPPO':                                                                                                                 # Unterscheidung zwischen RecurrentPPO und den anderen Algorithmen, da der Code sich stellenweise unterscheidet

            jm = 'by_id'
            
            obs = eval_env.reset(reset_mode=jm, journey_id=journey_eval_id, random_start=False)
            
            score = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                score += reward

            eval_env.close()

        else:

            jm = 'by_id'
        
            obs = eval_env.reset(reset_mode=jm, journey_id=journey_eval_id, random_start=False)
            lstm_states = None
            num_envs = 1
            episode_starts = np.ones((num_envs,), dtype=bool)
            
            score = 0
            done = False

            while not done:
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                episode_starts = done
                score += reward

            eval_env.close()

        rewards = score
        parking_error = abs(eval_env.journey.stopping_point[0] - eval_env.train.position)
        punctuality_error = abs(eval_env.accumulated_journey_time - eval_env.journey_time)

        rewards_counter += rewards                                                                                                                      # Variablen für die an Optuna übergebenen Evaluierungsergebnisse befüllen (Gesamtreward, Gesamtparkfehler und Gesamtzeitfehler)
        parking_error_counter += parking_error
        punctuality_error_counter += punctuality_error


    return rewards_counter, parking_error_counter, punctuality_error_counter                                                                            # Übergabe der Evaluierungsergebnisse an Optuna



class MyException(Exception):
    '''Empty Class for Custom Exception Messages'''
    pass

def main():

    global env                                                                                                                                          # env für die main-Funktion zugänglich machen

    '''
    Environment Creation
    '''

    if args.hyperparameter_tuning == True:                                                                                                              # Optuna-tuning ausführen
        
        storage_url = f"sqlite:///{hyperparameter_folder}/study{args.algo}.db"                                                                          # Speicherort der Optuna-Studie
        study = optuna.create_study(directions=["maximize", "minimize", "minimize"], sampler=optuna.samplers.TPESampler(multivariate=True), storage=storage_url, study_name=str(hyperparameter_folder),load_if_exists=True) # Erstellen der Optuna-Studie: Maximieren des ersten übergebenen Wertes und Minimieren der anderen beiden als Ziel; Verwenden des TPESamplers
        study.optimize(optimize_agent, timeout=(864000/2), show_progress_bar=True)                                                                      # Ausführen der Optuna-Studie (festlegen der Zeitdauer der Studie hier auf 5 Tage)
        for trial in study.best_trials:                                                                                                                 # print der besten trials (Empfehlung: die Ergebnisse können jedoch auch in Optuna Dashboard betrachtet werden)
            print("trial: ", trial.number, " best hyperparameters: ", trial.params)


        hyp_file = hyperparameter_folder / 'hyperparameters.txt'
        with open(hyp_file, 'w') as file:                                                                                                               # Schreiben der besten trials in ein file (Empfehlung: die Ergebnisse können jedoch auch in Optuna Dashboard betrachtet werden)
            file.write(' best hyperparameters\r\n')
            for trial in study.best_trials:
                file.write("              trial: "+str(trial.number)+" best hyperparameters: "+str(trial.params)+'\r\n')

            


    else:                                                                                                                                               # Wenn kein Optuna-tuning vorgenommen werden soll (also ein normales Training oder eine Evaluierung):
        '''
        Build Agent and Train
        '''
        # Read training folder from arguments. Training folder in which training results will be saved
        training_folder = Path(args.training_path) if args.train else None

        if args.algo == 'PPO': # PPO learning algorithm                                                                                                 # SB3-PPO (Festlegen der Hyperparameter und Erstellen des models)

            learning_rate       = 0.0003    # SB3 Standard: 0.0003
            n_steps             = 2048    # SB3 Standard: 2048
            batch_size          = 64    # SB3 Standard: 64
            n_epochs            = 10    # SB3 Standard: 10
            gamma               = 0.99    # SB3 Standard: 0.99
            gae_lambda          = 0.95    # SB3 Standard: 0.95
            clip_range          = 0.2    # SB3 Standard: 0.2
            ent_coef            = 0.01    # SB3 Standard: 0.0
            vf_coef             = 0.5    # SB3 Standard: 0.5
            max_grad_norm       = 0.5    # SB3 Standard: 0.5
            ortho_init          = False
            activation_fn       = torch.nn.ReLU
            net_arch            = dict(pi=[256,256], vf=[256,256])
            normalize_images    = False
            policy_kwargs       = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, normalize_images=normalize_images)
            time_steps          = 16e6

            agentPPO = PPO(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, device=device, verbose=0, 
                tensorboard_log=training_folder, policy_kwargs=policy_kwargs, seed=42) if args.train else PPO(policy="MultiInputPolicy", env=env, 
                learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, 
                ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, device=device, verbose=0, policy_kwargs=policy_kwargs)

            
        elif args.algo == 'DQN': # DQN learning algorithm                                                                                               # SB3-DQN (Festlegen der Hyperparameter und Erstellen des models)
            
            learning_rate       = 0.0003    # SB3 Standard: 0.0001
            buffer_size         = 10000    # SB3 Standard: 1000000
            learning_starts     = 5000    # SB3 Standard: 100
            batch_size          = 64    # SB3 Standard: 32
            tau                 = 1    # SB3 Standard: 1.0
            gamma               = 0.99    # SB3 Standard: 0.99
            train_freq          = 4    # SB3 Standard: 4
            gradient_steps      = 1    # SB3 Standard: 1
            target_update_interval = 1000    # SB3 Standard: 10000
            exploration_fraction = 0.75    # SB3 Standard: 0.1
            exploration_initial_eps = 1.0    # SB3 Standard: 1.0
            exploration_final_eps = 0.1    # SB3 Standard: 0.05
            max_grad_norm       = 10    # SB3 Standard: 10
            activation_fn       = torch.nn.ReLU
            net_arch            = [256, 256]
            normalize_images    = False
            policy_kwargs       = dict(activation_fn=activation_fn, net_arch=net_arch, normalize_images=normalize_images)
            time_steps          = 16e6

            agentDQN = DQN(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, 
                batch_size=batch_size, tau=tau, gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval, 
                exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, max_grad_norm=max_grad_norm, tensorboard_log=training_folder, 
                policy_kwargs=policy_kwargs, verbose=0, device=device, seed=42) if args.train else DQN(policy="MultiInputPolicy", env=env, 
                learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, 
                gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction, 
                exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, max_grad_norm=max_grad_norm, policy_kwargs=policy_kwargs, verbose=0, device=device)
            
        elif args.algo == 'A2C': # A2C learning algorithm                                                                                               # SB3-A2C (Festlegen der Hyperparameter und Erstellen des models)

            learning_rate       = 0.0021    # SB3 Standard: 0.0007
            n_steps             = 10    # SB3 Standard: 5
            gamma               = 0.99    # SB3 Standard: 0.99
            gae_lambda          = 0.95    # SB3 Standard: 1.0
            ent_coef            = 0.01    # SB3 Standard: 0.0
            vf_coef             = 0.5    # SB3 Standard: 0.5
            max_grad_norm       = 0.5    # SB3 Standard: 0.5
            rms_prop_eps        = 0.00001    # SB3 Standard: 0.00001
            use_rms_prop        = True    # SB3 Standard: True
            ortho_init          = False
            activation_fn       = torch.nn.ReLU
            net_arch            = dict(pi=[256,256], vf=[256,256])
            normalize_images    = False
            policy_kwargs       = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, normalize_images=normalize_images)
            time_steps          = 16e6

            agentA2C = A2C(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps,
                gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps, use_rms_prop=use_rms_prop, device=device, verbose=0, 
                tensorboard_log=training_folder, policy_kwargs=policy_kwargs, seed=42) if args.train else A2C(policy="MultiInputPolicy", env=env, 
                learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda, 
                ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, rms_prop_eps=rms_prop_eps, use_rms_prop=use_rms_prop, device=device, verbose=0, policy_kwargs=policy_kwargs)
            
        elif args.algo == 'RecurrentPPO': # RecurrentPPO learning algorithm                                                                             # SB3-RecurrentPPO (Festlegen der Hyperparameter und Erstellen des models)

            learning_rate       = 0.0003    # SB3 Standard: 0.0003
            n_steps             = 2048    # SB3 Standard: 128
            batch_size          = 128    # SB3 Standard: 128
            n_epochs            = 10    # SB3 Standard: 10
            gamma               = 0.99    # SB3 Standard: 0.99
            gae_lambda          = 0.95    # SB3 Standard: 0.95
            clip_range          = 0.2    # SB3 Standard: 0.2
            ent_coef            = 0.0    # SB3 Standard: 0.0
            vf_coef             = 0.5    # SB3 Standard: 0.5
            max_grad_norm       = 0.5    # SB3 Standard: 0.5
            ortho_init          = False
            activation_fn       = torch.nn.ReLU
            net_arch            = dict(pi=[256,256], vf=[256,256])
            lstm_hidden_size    = 128    # SB3 Standard: 256
            n_lstm_layers       = 1    # SB3 Standard: 1
            shared_lstm         = False    # SB3 Standard: False
            enable_critic_lstm  = True    # SB3 Standard: True
            normalize_images    = False
            policy_kwargs = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, lstm_hidden_size=lstm_hidden_size, n_lstm_layers=n_lstm_layers, shared_lstm=shared_lstm, enable_critic_lstm=enable_critic_lstm, normalize_images=normalize_images)
            time_steps          = 16e6

            agentRecurrentPPO = RecurrentPPO(policy="MultiInputLstmPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, device=device, verbose=0, 
                tensorboard_log=training_folder, policy_kwargs=policy_kwargs, seed=42) if args.train else RecurrentPPO(policy="MultiInputLstmPolicy", env=env, 
                learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, clip_range=clip_range, 
                ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, device=device, verbose=0, policy_kwargs=policy_kwargs)
            
        elif args.algo == 'QRDQN': # QR-DQN learning algorithm                                                                                          # SB3-QRDQN (Festlegen der Hyperparameter und Erstellen des models)

            learning_rate       = 5e-05    # SB3 Standard: 5e-05
            buffer_size         = 1000000    # SB3 Standard: 1000000
            learning_starts     = 100    # SB3 Standard: 100
            batch_size          = 32    # SB3 Standard: 32
            tau                 = 1.0    # SB3 Standard: 1.0
            gamma               = 0.99    # SB3 Standard: 0.99
            train_freq          = 4    # SB3 Standard: 4
            gradient_steps      = 1    # SB3 Standard: 1
            target_update_interval = 10000    # SB3 Standard: 10000
            exploration_fraction = 0.5    # SB3 Standard: 0.005
            exploration_initial_eps = 1.0    # SB3 Standard: 1.0
            exploration_final_eps = 0.037    # SB3 Standard: 0.01
            max_grad_norm       = None    # SB3 Standard: None
            n_quantiles         = 200    # SB3 Standard: 200
            activation_fn       = torch.nn.ReLU
            net_arch            = [256, 256]
            normalize_images    = False
            policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch, n_quantiles=n_quantiles, normalize_images=normalize_images)
            time_steps          = 16e6

            agentQRDQN = QRDQN(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts,
                batch_size=batch_size, tau=tau, gamma=gamma, train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction, exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, max_grad_norm=max_grad_norm, 
                device=device, verbose=0, tensorboard_log=training_folder, policy_kwargs=policy_kwargs, seed=42) if args.train else QRDQN(policy="MultiInputPolicy", env=env, 
                learning_rate=learning_rate, buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma, train_freq=train_freq, 
                gradient_steps=gradient_steps, target_update_interval=target_update_interval, exploration_fraction=exploration_fraction, 
                exploration_initial_eps=exploration_initial_eps, exploration_final_eps=exploration_final_eps, max_grad_norm=max_grad_norm, 
                device=device, verbose=0, policy_kwargs=policy_kwargs)

            
        elif args.algo == 'TRPO': # TRPO learning algorithm                                                                                             # SB3-TRPO (Festlegen der Hyperparameter und Erstellen des models)

            learning_rate       = 0.0003    # SB3 Standard: 0.001
            n_steps             = 2048    # SB3 Standard: 2048
            batch_size          = 64    # SB3 Standard: 128
            gamma               = 0.99    # SB3 Standard: 0.99
            cg_max_steps        = 15    # SB3 Standard: 15
            cg_damping          = 0.1    # SB3 Standard: 0.1
            line_search_shrinking_factor = 0.8    # SB3 Standard: 0.8
            line_search_max_iter = 10    # SB3 Standard: 10
            n_critic_updates    = 10    # SB3 Standard: 10
            gae_lambda          = 0.95    # SB3 Standard: 0.95
            sub_sampling_factor = 1    # SB3 Standard: 1
            ortho_init          = False
            activation_fn       = torch.nn.ReLU
            net_arch            = dict(pi=[256,256], vf=[256,256])
            normalize_images    = False
            policy_kwargs = dict(activation_fn=activation_fn, ortho_init=ortho_init, net_arch=net_arch, normalize_images=normalize_images)
            time_steps          = 16e6

            agentTRPO = TRPO(policy="MultiInputPolicy", env=env, learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, gamma=gamma,
                cg_max_steps=cg_max_steps, cg_damping=cg_damping, line_search_shrinking_factor=line_search_shrinking_factor, 
                line_search_max_iter=line_search_max_iter, n_critic_updates=n_critic_updates, gae_lambda=gae_lambda, sub_sampling_factor=sub_sampling_factor, device=device, 
                verbose=0, tensorboard_log=training_folder, policy_kwargs=policy_kwargs, seed=42) if args.train else TRPO(policy="MultiInputPolicy", env=env, 
                learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size, gamma=gamma, cg_max_steps=cg_max_steps, cg_damping=cg_damping, 
                line_search_shrinking_factor=line_search_shrinking_factor, line_search_max_iter=line_search_max_iter, n_critic_updates=n_critic_updates, 
                gae_lambda=gae_lambda, sub_sampling_factor=sub_sampling_factor, device=device, verbose=0, policy_kwargs=policy_kwargs)


        else:
            raise MyException("Undefined or unknown algorithm.")  # Use 'PPO' or 'DDQN'.")

        if args.train:  # Training Mode activated

            print('INFO: Training Agent')

            if args.algo == 'PPO':                                                                                                                      # PPO: Schreiben der Hyperparameter in ein file; Trainingsschleife (Lernen, Tensorboard-logging und Speichern der angelernten models)

                print('INFO: Training with PPO algorithm')

                hyp_file = training_folder / 'hyperparameters.txt'
                with open(hyp_file, 'w') as file:
                    file.write('PPO Hyperparameters\n')
                    file.write('    learning_rate   = '+str(learning_rate)+'\n')
                    file.write('    n_steps         = '+str(n_steps)+'\n')
                    file.write('    batch_size      = '+str(batch_size)+'\n')
                    file.write('    n_epochs        = '+str(n_epochs)+'\n')
                    file.write('    gamma           = '+str(gamma)+'\n')
                    file.write('    gae_lambda      = '+str(gae_lambda)+'\n')
                    file.write('    clip_range      = '+str(clip_range)+'\n')
                    file.write('    ent_coef        = '+str(ent_coef)+'\n')
                    file.write('    vf_coef         = '+str(vf_coef)+'\n')
                    file.write('    max_grad_norm   = '+str(max_grad_norm)+'\n')
                    file.write('    time_steps      = '+str(time_steps)+'\n')
                    
                print('INFO: Hyperparameters saved at '+str(hyp_file))
                print('INFO: Training started ...')

                num_iters = 10
                for iter in tqdm(range(num_iters)):
                    agentPPO.learn(total_timesteps=time_steps/num_iters,tb_log_name='PPO_tensorboard_log'+str(iter), reset_num_timesteps=False, progress_bar=True)
                    agentPPO.save(f"{training_folder}/{'agent'+str(iter)}")

                print('INFO: Training ended ...')

            elif args.algo == 'DQN':                                                                                                                    # DQN: Schreiben der Hyperparameter in ein file; Trainingsschleife (Lernen, Tensorboard-logging und Speichern der angelernten models)

                print('INFO: Training with PPO algorithm')

                hyp_file = training_folder / 'hyperparameters.txt'
                with open(hyp_file, 'w') as file:
                    file.write('DQN Hyperparameters\n')
                    file.write('    learning_rate               = '+str(learning_rate)+'\n')
                    file.write('    buffer_size                 = '+str(buffer_size)+'\n')
                    file.write('    learning_starts             = '+str(learning_starts)+'\n')
                    file.write('    batch_size                  = '+str(batch_size)+'\n')
                    file.write('    tau                         = '+str(tau)+'\n')
                    file.write('    gamma                       = '+str(gamma)+'\n')
                    file.write('    train_freq                  = '+str(train_freq)+'\n')
                    file.write('    gradient_steps              = '+str(gradient_steps)+'\n')
                    file.write('    target_update_interval      = '+str(target_update_interval)+'\n')
                    file.write('    exploration_fraction        = '+str(exploration_fraction)+'\n')
                    file.write('    exploration_initial_eps     = '+str(exploration_initial_eps)+'\n')
                    file.write('    exploration_final_eps       = '+str(exploration_final_eps)+'\n')
                    file.write('    max_grad_norm               = '+str(max_grad_norm)+'\n')
                    file.write('    time_steps                  = '+str(time_steps)+'\n')

                print('INFO: Hyperparameters saved at '+str(hyp_file))
                print('INFO: Training started ...')

                num_iters = 10
                for iter in tqdm(range(num_iters)):
                    agentDQN.learn(total_timesteps=time_steps/num_iters,tb_log_name='DQN_tensorboard_log'+str(iter), reset_num_timesteps=False, progress_bar=True)
                    agentDQN.save(f"{training_folder}/{'agent'+str(iter)}")

                print('INFO: Training ended ...')

            elif args.algo == 'A2C':                                                                                                                    # A2C: Schreiben der Hyperparameter in ein file; Trainingsschleife (Lernen, Tensorboard-logging und Speichern der angelernten models)

                print('INFO: Training with A2C algorithm')

                hyp_file = training_folder / 'hyperparameters.txt'
                with open(hyp_file, 'w') as file:
                    file.write('A2C Hyperparameters\n')
                    file.write('    learning_rate   = '+str(learning_rate)+'\n')
                    file.write('    n_steps         = '+str(n_steps)+'\n')
                    file.write('    gamma           = '+str(gamma)+'\n')
                    file.write('    gae_lambda      = '+str(gae_lambda)+'\n')
                    file.write('    ent_coef        = '+str(ent_coef)+'\n')
                    file.write('    vf_coef         = '+str(vf_coef)+'\n')
                    file.write('    max_grad_norm   = '+str(max_grad_norm)+'\n')
                    file.write('    rms_prop_eps    = '+str(rms_prop_eps)+'\n')
                    file.write('    use_rms_prop    = '+str(use_rms_prop)+'\n')
                    file.write('    time_steps      = '+str(time_steps)+'\n')
                    
                print('INFO: Hyperparameters saved at '+str(hyp_file))
                print('INFO: Training started ...')

                num_iters = 10
                for iter in tqdm(range(num_iters)):
                    agentA2C.learn(total_timesteps=time_steps/num_iters,tb_log_name='A2C_tensorboard_log'+str(iter), reset_num_timesteps=False, progress_bar=True)
                    agentA2C.save(f"{training_folder}/{'agent'+str(iter)}")

                print('INFO: Training ended ...')

            elif args.algo == 'RecurrentPPO':                                                                                                           # RecurrentPPO: Schreiben der Hyperparameter in ein file; Trainingsschleife (Lernen, Tensorboard-logging und Speichern der angelernten models)

                print('INFO: Training with RecurrentPPO algorithm')

                hyp_file = training_folder / 'hyperparameters.txt'
                with open(hyp_file, 'w') as file:
                    file.write('RecurrentPPO Hyperparameters\n')
                    file.write('    learning_rate       = '+str(learning_rate)+'\n')
                    file.write('    n_steps             = '+str(n_steps)+'\n')
                    file.write('    batch_size          = '+str(batch_size)+'\n')
                    file.write('    n_epochs            = '+str(n_epochs)+'\n')
                    file.write('    gamma               = '+str(gamma)+'\n')
                    file.write('    gae_lambda          = '+str(gae_lambda)+'\n')
                    file.write('    clip_range          = '+str(clip_range)+'\n')
                    file.write('    ent_coef            = '+str(ent_coef)+'\n')
                    file.write('    vf_coef             = '+str(vf_coef)+'\n')
                    file.write('    max_grad_norm       = '+str(max_grad_norm)+'\n')
                    file.write('    lstm_hidden_size    = '+str(lstm_hidden_size)+'\n')
                    file.write('    n_lstm_layers       = '+str(n_lstm_layers)+'\n')
                    file.write('    shared_lstm         = '+str(shared_lstm)+'\n')
                    file.write('    enable_critic_lstm  = '+str(enable_critic_lstm)+'\n')
                    file.write('    time_steps          = '+str(time_steps)+'\n')

                    
                print('INFO: Hyperparameters saved at '+str(hyp_file))
                print('INFO: Training started ...')

                num_iters = 10
                for iter in tqdm(range(num_iters)):
                    agentRecurrentPPO.learn(total_timesteps=time_steps/num_iters,tb_log_name='RecurrentPPO_tensorboard_log'+str(iter), reset_num_timesteps=False, progress_bar=True)
                    agentRecurrentPPO.save(f"{training_folder}/{'agent'+str(iter)}")

                print('INFO: Training ended ...')

            elif args.algo == 'QRDQN':                                                                                                                  # QRDQN: Schreiben der Hyperparameter in ein file; Trainingsschleife (Lernen, Tensorboard-logging und Speichern der angelernten models)

                print('INFO: Training with QRDQN algorithm')

                hyp_file = training_folder / 'hyperparameters.txt'
                with open(hyp_file, 'w') as file:
                    file.write('QRDQN Hyperparameters\n')
                    file.write('    learning_rate           = '+str(learning_rate)+'\n')
                    file.write('    buffer_size             = '+str(buffer_size)+'\n')
                    file.write('    learning_starts         = '+str(learning_starts)+'\n')
                    file.write('    batch_size              = '+str(batch_size)+'\n')
                    file.write('    tau                     = '+str(tau)+'\n')
                    file.write('    gamma                   = '+str(gamma)+'\n')
                    file.write('    train_freq              = '+str(train_freq)+'\n')
                    file.write('    gradient_steps          = '+str(gradient_steps)+'\n')
                    file.write('    target_update_interval  = '+str(target_update_interval)+'\n')
                    file.write('    exploration_fraction    = '+str(exploration_fraction)+'\n')
                    file.write('    exploration_initial_eps = '+str(exploration_initial_eps)+'\n')
                    file.write('    exploration_final_eps   = '+str(exploration_final_eps)+'\n')
                    file.write('    max_grad_norm           = '+str(max_grad_norm)+'\n')
                    file.write('    n_quantiles             = '+str(n_quantiles)+'\n')
                    file.write('    time_steps              = '+str(time_steps)+'\n')

                    
                print('INFO: Hyperparameters saved at '+str(hyp_file))
                print('INFO: Training started ...')

                num_iters = 10
                for iter in tqdm(range(num_iters)):
                    agentQRDQN.learn(total_timesteps=time_steps/num_iters,tb_log_name='QRDQN_tensorboard_log'+str(iter), reset_num_timesteps=False, progress_bar=True)
                    agentQRDQN.save(f"{training_folder}/{'agent'+str(iter)}")

                print('INFO: Training ended ...')

            elif args.algo == 'TRPO':                                                                                                                   # TRPO: Schreiben der Hyperparameter in ein file; Trainingsschleife (Lernen, Tensorboard-logging und Speichern der angelernten models)

                print('INFO: Training with TRPO algorithm')

                hyp_file = training_folder / 'hyperparameters.txt'
                with open(hyp_file, 'w') as file:
                    file.write('TRPO Hyperparameters\n')
                    file.write('    learning_rate                   = '+str(learning_rate)+'\n')
                    file.write('    n_steps                         = '+str(n_steps)+'\n')
                    file.write('    batch_size                      = '+str(batch_size)+'\n')
                    file.write('    gamma                           = '+str(gamma)+'\n')
                    file.write('    cg_max_steps                    = '+str(cg_max_steps)+'\n')
                    file.write('    cg_damping                      = '+str(cg_damping)+'\n')
                    file.write('    line_search_shrinking_factor    = '+str(line_search_shrinking_factor)+'\n')
                    file.write('    line_search_max_iter            = '+str(line_search_max_iter)+'\n')
                    file.write('    n_critic_updates                = '+str(n_critic_updates)+'\n')
                    file.write('    gae_lambda                      = '+str(gae_lambda)+'\n')
                    file.write('    sub_sampling_factor             = '+str(sub_sampling_factor)+'\n')
                    file.write('    time_steps                      = '+str(time_steps)+'\n')

                    
                print('INFO: Hyperparameters saved at '+str(hyp_file))
                print('INFO: Training started ...')

                num_iters = 10
                for iter in tqdm(range(num_iters)):
                    agentTRPO.learn(total_timesteps=time_steps/num_iters,tb_log_name='TRPO_tensorboard_log'+str(iter), reset_num_timesteps=False, progress_bar=True)
                    agentTRPO.save(f"{training_folder}/{'agent'+str(iter)}")

                print('INFO: Training ended ...')


        '''
        Evaluate
        '''

        if not args.train:

            evaluation_folder = Path(args.evaluation_path)

            if args.algo == 'RecurrentPPO':                                                                                                             # Unterscheidung zwischen RecurrentPPO und den anderen Algorithmen für den Code der Evaluierung

                positions_arr = []
                actions_arr = []
                speeds_arr = []
                accelerations_arr = []
                jerks_arr = []
                punctuality_arr = []
                safety_arr = []
                energy_arr = []

                env = gym.make('gym_train:train-v0')

                if args.journey_mode == 0:
                    jm = 'arbitrary'
                elif args.journey_mode == 1:
                    jm = 'from_xml'
                elif args.journey_mode == 2:
                    jm = 'by_id'
                else:
                    raise MyException("Undefined journey mode. Expected 0, 1 0r 2.")
                
                obs = env.reset(reset_mode=jm, journey_id=args.journey_id, random_start=False)
                agent_name = args.agent_name + '.zip'
                model_path = evaluation_folder / agent_name
                model = RecurrentPPO.load(model_path, env=env)                                                                                          # Laden des models
                lstm_states = None
                num_envs = 1
                episode_starts = np.ones((num_envs,), dtype=bool)
                
                env.journey.print_journey()
                collect_metrics(env, 0, jerks_arr, actions_arr, positions_arr, speeds_arr, \
                                accelerations_arr, punctuality_arr, safety_arr, energy_arr)
                
                score = 0
                done = False
                step = 0                                                                                                                                # neue Variable, um steps zu zählen
                exceededspeedlimits = False                                                                                                             # neue Variable, um festzustellen, ob die Geschwindigkeitsbegrenzung mindestens einmal überschritten wurde
                exceededspeedlimitsstepcounter = 0                                                                                                      # neue Variable, um festzustellen für wie viele steps die Geschwindigkeitsbegrenzung überschritten wurde
                exceededspeedlimitsstep = 0                                                                                                             # neue Variable, um festzustellen bei welchem step die Geschwindigkeitsbegrenzung erstmalig überschritten wurde
                exceededspeedlimitsspeed = 0                                                                                                            # neue Variable, um festzustellen mit welchem speed die Geschwindigkeitsbegrenzung erstmalig überschritten wurde
                exceededjerklimit = False                                                                                                               # neue Variable, um festzustellen, ob das Jerk-Limit mindestens einmal überschritten wurde
                exceededjerklimitstepcounter = 0                                                                                                        # neue Variable, um festzustellen für wie viele steps das Jerk-Limit überschritten wurde
                exceededjerklimitstep = 0                                                                                                               # neue Variable, um festzustellen bei welchem step das Jerk-Limit erstmalig überschritten wurde
                exceededjerklimitjerk = 0                                                                                                               # neue Variable, um festzustellen mit welchem Jerkwert das Jerk-Limit erstmalig überschritten wurde
                safety_flag = True
                comfort_flag = True

                reward = 0                                                                                                                              # Initialisieren der Variable "reward" für die Auswertungsfunktionen
                action = 5                                                                                                                              # Initialisieren der Variable "action" für die Auswertungsfunktionen, 5 für RTB = 0


                print_evaluation(env, evaluation_folder, step, score, reward, action, obs)                                                              # Schreiben der Evaluierungsdetails in files


                while not done:
                    if args.render:
                        env.render()
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    episode_starts = done
                    collect_metrics(env, env.translator(action), jerks_arr, actions_arr, \
                                    positions_arr, speeds_arr, accelerations_arr, punctuality_arr, \
                                    safety_arr, energy_arr)
                    score += reward
                    step += 1                                                                                                                           # Step-Zähler: +1

                    if env.r_safety == 0.0:
                        safety_flag = True
                    if env.r_safety < 0.0 and safety_flag == True:                                                                                      # neue Bedingung, um festzustellen, ob die Geschwindigkeitsbegrenzung mindestens einmal überschritten wurde, und wie viele Male insgesamt
                        if exceededspeedlimits == False:                                                                                                # ^
                            exceededspeedlimitsstep = step                                                                                              # ^
                            exceededspeedlimitsspeed = env.train.speed * 3.6                                                                            # ^
                            exceededspeedlimits = True                                                                                                  # ^
                        exceededspeedlimitsstepcounter += 1                                                                                             # ^
                        safety_flag = False

                    if env.r_comfort == 0.0:
                        comfort_flag = True
                    if env.r_comfort < 0.0 and comfort_flag == True:                                                                                    # neue Bedingung, um festzustellen, ob das Jerk-Limit mindestens einmal überschritten wurde, und wie viele Male insgesamt
                        if exceededjerklimit == False:                                                                                                  # ^
                            exceededjerklimitstep = step                                                                                                # ^
                            exceededjerklimitjerk = env.max_jerk                                                                                        # ^
                            exceededjerklimit = True                                                                                                    # ^
                        exceededjerklimitstepcounter += 1                                                                                               # ^
                        comfort_flag = False

                    print_evaluation(env, evaluation_folder, step, score, reward, action, obs)                                                          # Schreiben der Evaluierungsdetails in files


                env.close()

            else:

                positions_arr = []
                actions_arr = []
                speeds_arr = []
                accelerations_arr = []
                jerks_arr = []
                punctuality_arr = []
                safety_arr = []
                energy_arr = []

                env = gym.make('gym_train:train-v0')

                if args.journey_mode == 0:
                    jm = 'arbitrary'
                elif args.journey_mode == 1:
                    jm = 'from_xml'
                elif args.journey_mode == 2:
                    jm = 'by_id'
                else:
                    raise MyException("Undefined journey mode. Expected 0, 1 0r 2.")
                
                obs = env.reset(reset_mode=jm, journey_id=args.journey_id, random_start=False)
                agent_name = args.agent_name + '.zip'
                model_path = evaluation_folder / agent_name

                if args.algo == 'PPO':                                                                                                                  # Laden des models (je nach Algorithmus)
                    model = PPO.load(model_path, env=env)
                elif args.algo == 'DQN':
                    model = DQN.load(model_path, env=env)
                elif args.algo == 'A2C':
                    model = A2C.load(model_path, env=env)
                elif args.algo == 'QRDQN':
                    model = QRDQN.load(model_path, env=env)
                elif args.algo == 'TRPO':
                    model = TRPO.load(model_path, env=env)

                
                env.journey.print_journey()
                collect_metrics(env, 0, jerks_arr, actions_arr, positions_arr, speeds_arr, \
                                accelerations_arr, punctuality_arr, safety_arr, energy_arr)
                
                score = 0
                done = False
                step = 0                                                                                                                                # neue Variable, um steps zu zählen
                exceededspeedlimits = False                                                                                                             # neue Variable, um festzustellen, ob die Geschwindigkeitsbegrenzung mindestens einmal überschritten wurde
                exceededspeedlimitsstepcounter = 0                                                                                                      # neue Variable, um festzustellen für wie viele steps die Geschwindigkeitsbegrenzung überschritten wurde
                exceededspeedlimitsstep = 0                                                                                                             # neue Variable, um festzustellen bei welchem step die Geschwindigkeitsbegrenzung erstmalig überschritten wurde
                exceededspeedlimitsspeed = 0                                                                                                            # neue Variable, um festzustellen mit welchem speed die Geschwindigkeitsbegrenzung erstmalig überschritten wurde
                exceededjerklimit = False                                                                                                               # neue Variable, um festzustellen, ob das Jerk-Limit mindestens einmal überschritten wurde
                exceededjerklimitstepcounter = 0                                                                                                        # neue Variable, um festzustellen für wie viele steps das Jerk-Limit überschritten wurde
                exceededjerklimitstep = 0                                                                                                               # neue Variable, um festzustellen bei welchem step das Jerk-Limit erstmalig überschritten wurde
                exceededjerklimitjerk = 0                                                                                                               # neue Variable, um festzustellen mit welchem Jerkwert das Jerk-Limit erstmalig überschritten wurde
                safety_flag = True
                comfort_flag = True

                reward = 0                                                                                                                              # Initialisieren der Variable für die Auswertungsfunktionen
                action = 5                                                                                                                              # Initialisieren der Variable für die Auswertungsfunktionen, 5 für RTB = 0


                print_evaluation(env, evaluation_folder, step, score, reward, action, obs)                                                              # Schreiben der Evaluierungsdetails in files


                while not done:
                    if args.render:
                        env.render()
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _ = env.step(action)
                    collect_metrics(env, env.translator(action), jerks_arr, actions_arr, \
                                    positions_arr, speeds_arr, accelerations_arr, punctuality_arr, \
                                    safety_arr, energy_arr)
                    score += reward
                    step += 1                                                                                                                           # Step-Zähler: +1

                    if env.r_safety == 0.0:
                        safety_flag = True
                    if env.r_safety < 0.0 and safety_flag == True:                                                                                      # neue Bedingung, um festzustellen, ob die Geschwindigkeitsbegrenzung mindestens einmal überschritten wurde, und wie viele Male insgesamt
                        if exceededspeedlimits == False:                                                                                                # ^
                            exceededspeedlimitsstep = step                                                                                              # ^
                            exceededspeedlimitsspeed = env.train.speed * 3.6                                                                            # ^
                            exceededspeedlimits = True                                                                                                  # ^
                        exceededspeedlimitsstepcounter += 1                                                                                             # ^
                        safety_flag = False

                    if env.r_comfort == 0.0:
                        comfort_flag = True
                    if env.r_comfort < 0.0 and comfort_flag == True:                                                                                    # neue Bedingung, um festzustellen, ob das Jerk-Limit mindestens einmal überschritten wurde, und wie viele Male insgesamt
                        if exceededjerklimit == False:                                                                                                  # ^
                            exceededjerklimitstep = step                                                                                                # ^
                            exceededjerklimitjerk = env.max_jerk                                                                                        # ^
                            exceededjerklimit = True                                                                                                    # ^
                        exceededjerklimitstepcounter += 1                                                                                               # ^
                        comfort_flag = False

                    print_evaluation(env, evaluation_folder, step, score, reward, action, obs)                                                          # Schreiben der Evaluierungsdetails in files


                env.close()
            
            parking_error = abs(np.round(env.journey.stopping_point[0] - env.train.position, 4))

            if (env.journey.stopping_point[0] - env.train.position) < 0:
                parking_error_direction = " after perfect position"
            else:
                parking_error_direction = " before perfect position"

            print('Reason for Quit          = ', env.reason_for_quit, ' at position: ', env.train.position)                                             # ergänzt: print: Grund für das Beenden der episode
            print('Total Score              = ', np.round(score, 4))
            print('Parking Error            = ', parking_error, ' m ', parking_error_direction)                                                         # angepasst: Rundung von 2 auf 4
            print('Punctuality Error        = ', np.round(env.accumulated_journey_time - env.journey_time, 1), ' s')                                    # angepasst: von "np.round(env.delta_time_left, 1)" auf "np.round(env.accumulated_journey_time - env.journey_time, 1)"
            print('Exceeded Speedlimits     = ', exceededspeedlimits, ' for ', exceededspeedlimitsstepcounter, ' times, first: at step: ', exceededspeedlimitsstep, 
                ' with speed: ', exceededspeedlimitsspeed)                                                                                              # ergänzt: print: eindeutige Aussage, ob die Geschwindigkeitsbegrenzung mindestens einmal überschritten wurde (und bei welchem step mit welchem speed), und wie viele Male insgesamt
            print('Exceeded Jerklimit       = ', exceededjerklimit, ' for ', exceededjerklimitstepcounter, ' times, first: at step: ', exceededjerklimitstep, 
                ' with jerk: ', exceededjerklimitjerk)                                                                                                  # ergänzt: print: eindeutige Aussage, ob der maximale erlaubte Jerk-Wert mindestens einmal überschritten wurde (und bei welchem step mit welchem jerk), und wie viele Male insgesamt
            print('Cummulated Energy        = ', np.round(env.cummulated_energy, 4), ' m^2/s^2')                                                        # ergänzt: print: die insgesamt "angesammelte" Energie
            plot_name = args.agent_name + '.png'
            plot_path = evaluation_folder / plot_name
            plot_metrics(env, plot_path, positions_arr, speeds_arr, accelerations_arr, \
                        actions_arr, punctuality_arr, jerks_arr, energy_arr)
            

            evaluationpath = evaluation_folder / (args.agent_name + 'evaluationdetails.txt')                                                            # zusätzliche Augabe der obigen Evaluierungs-Daten in einem .txt-file
            with open(evaluationpath, 'w') as file:
                file.write('Evaluationdetails\n')
                file.write('    Reason for Quit          = '+str(env.reason_for_quit)+' at position: '+ str(env.train.position)+'\n')
                file.write('    Total Score              = '+str(np.round(score, 4))+'\n')
                file.write('    Parking Error            = '+str(parking_error)+' m '+str(parking_error_direction)+'\n')
                file.write('    Punctuality Error        = '+str(np.round(env.accumulated_journey_time - env.journey_time, 1))+' s'+'\n')
                file.write('    Exceeded Speedlimits     = '+str(exceededspeedlimits)+' for '+str(exceededspeedlimitsstepcounter)+' times, first: at step: '+str(exceededspeedlimitsstep)+' with speed: '+str(exceededspeedlimitsspeed)+'\n')
                file.write('    Exceeded Jerklimit       = '+str(exceededjerklimit)+' for '+str(exceededjerklimitstepcounter)+' times, first: at step: '+str(exceededjerklimitstep)+' with jerk: '+str(exceededjerklimitjerk)+'\n')
                file.write('    Cummulated Energy        = '+str(np.round(env.cummulated_energy, 4))+' m^2/s^2'+'\n')

            
            evaluationpath = evaluation_folder / (args.agent_name + 'evaluationdetails.csv')                                                            # Schreiben einer .csv Datei mit den in der command-line ausgegebenen Daten
            dataevaluation = {'startingpoint': ("Starting Point : ", env.journey.starting_point[2]), 'startingposition': ("Starting Position : ", env.journey.starting_point[0]), 
                            'stoppingpoint': ("Stopping Point : ", env.journey.stopping_point[2]), 'stoppingposition': ("Stopping Position : ", env.journey.stopping_point[0]), 
                            'reasonforquit': ('Reason for Quit          = ', env.reason_for_quit, ' at position: ', env.train.position), 'totalscore': ('Total Score              = ', np.round(score, 4)), 
                            'parkingerror': ('Parking Error            = ', parking_error, ' m ', parking_error_direction), 
                            'punctualityerror': ('Punctuality Error        = ', np.round(env.accumulated_journey_time - env.journey_time, 1), ' s'), 
                            'exceededspeedlimits': ('Exceeded Speedlimits     = ', exceededspeedlimits, ' for ', exceededspeedlimitsstepcounter, ' times, first: at step: ', exceededspeedlimitsstep, 'with speed: ', exceededspeedlimitsspeed), 
                            'exceededjerklimit': ('Exceeded Jerklimit       = ', exceededjerklimit, ' for ', exceededjerklimitstepcounter, ' times, first: at step: ', exceededjerklimitstep, 'with jerk: ', exceededjerklimitjerk), 
                            'cummulatedenergy': ('Cummulated Energy        = ', np.round(env.cummulated_energy, 4), ' m^2/s^2')}
            with open(evaluationpath, mode='a', newline='') as file:
                writerevaluation = csv.writer(file)
                writerevaluation.writerow([dataevaluation['startingpoint'], dataevaluation['startingposition'], dataevaluation['stoppingpoint'], dataevaluation['stoppingposition'],
                                            dataevaluation['reasonforquit'], dataevaluation['totalscore'], dataevaluation['parkingerror'], dataevaluation['punctualityerror'],
                                            dataevaluation['exceededspeedlimits'], dataevaluation['exceededjerklimit'], dataevaluation['cummulatedenergy'], ''])



def print_evaluation(env:TrainEnv, evaluation_folder, step, score, reward, action, obs):                                                                # Funktion: Schreiben der Evaluierungsdetails in files

    rewardsperpositionandsteppath = evaluation_folder / (args.agent_name + 'rewards.csv')                                                               # Schreiben einer .csv Datei mit step, position, score, reward und allen einzelrewards
    datarewardsperpositionandstep = {'step': step, 'position': env.train.position, 'score': score, 'reward': reward, 'safety': env.r_safety,
                                      'comfort': env.r_comfort, 'energy': env.r_energy, 'guide': env.r_guide, 'parking': env.r_parking, 'punctuality': env.r_punctuality}
    with open(rewardsperpositionandsteppath, mode='a', newline='') as file:
        writerrewardsperpositionandstep = csv.writer(file)
        writerrewardsperpositionandstep.writerow([datarewardsperpositionandstep['step'], datarewardsperpositionandstep['position'], datarewardsperpositionandstep['score'],
                                                   datarewardsperpositionandstep['reward'], datarewardsperpositionandstep['safety'], datarewardsperpositionandstep['comfort'],
                                                   datarewardsperpositionandstep['energy'], datarewardsperpositionandstep['guide'], datarewardsperpositionandstep['parking'],
                                                   datarewardsperpositionandstep['punctuality'], ''])

    unscaledobsperpositionandsteppath = evaluation_folder / (args.agent_name + 'unscaledobservation.csv')                                               # Schreiben einer .csv Datei mit step, position, und den unskalierten observations
    dataunscaledobsperpositionandstep = {'step': step, 'position': env.train.position, 'action': action, 'currentspeed': env.train.speed,
                                          'distancetodestination': env.distance_to_destination, 'acceleration': env.train.acceleration,
                                          'timeerror': env.delta_time_left, 'jerk': env.max_jerk, 'distancecovered': env.distance_covered,
                                          'currentgradient': env.current_gradient, 'distancenextgradient': env.distance_next_gradient,
                                          'nextgradient': env.next_gradient, 'futurespeedlimits': env.future_speed_limit_segments}
    with open(unscaledobsperpositionandsteppath, mode='a', newline='') as file:
        writerunscaledobsperpositionandstep = csv.writer(file)
        writerunscaledobsperpositionandstep.writerow([dataunscaledobsperpositionandstep['step'], dataunscaledobsperpositionandstep['position'], 
                                                       dataunscaledobsperpositionandstep['action'], dataunscaledobsperpositionandstep['currentspeed'], 
                                                       dataunscaledobsperpositionandstep['distancetodestination'], dataunscaledobsperpositionandstep['acceleration'], 
                                                       dataunscaledobsperpositionandstep['timeerror'], dataunscaledobsperpositionandstep['jerk'], 
                                                       dataunscaledobsperpositionandstep['distancecovered'], dataunscaledobsperpositionandstep['currentgradient'],
                                                       dataunscaledobsperpositionandstep['distancenextgradient'], dataunscaledobsperpositionandstep['nextgradient'],
                                                       dataunscaledobsperpositionandstep['futurespeedlimits'], ''])

    obsperpositionandsteppath = evaluation_folder / (args.agent_name + 'observation.csv')                                                               # Schreiben einer .csv Datei mit step, position, und den observations
    dataobsperpositionandstep = {'step': step, 'position': env.train.position, 'action': obs["1"][0], 'currentspeed': obs["1"][1], 
                                    'distancetodestination': obs["1"][2], 'acceleration': obs["1"][3], 'timeerror': obs["1"][4], 'jerk': obs["1"][5], 
                                    'distancecovered': obs["1"][6], 'currentgradient': obs["1"][7],
                                    'distancenextgradient': obs["1"][8], 'nextgradient': obs["1"][9],
                                    'futurespeedlimits': obs["speedlimits"]}
    with open(obsperpositionandsteppath, mode='a', newline='') as file:
        writerobsperpositionandstep = csv.writer(file)
        writerobsperpositionandstep.writerow([dataobsperpositionandstep['step'], dataobsperpositionandstep['position'], 
                                                dataobsperpositionandstep['action'], dataobsperpositionandstep['currentspeed'], 
                                                dataobsperpositionandstep['distancetodestination'], dataobsperpositionandstep['acceleration'], 
                                                dataobsperpositionandstep['timeerror'], dataobsperpositionandstep['jerk'], 
                                                dataobsperpositionandstep['distancecovered'], dataobsperpositionandstep['currentgradient'],
                                                dataobsperpositionandstep['distancenextgradient'], dataobsperpositionandstep['nextgradient'], 
                                                dataobsperpositionandstep['futurespeedlimits'], ''])


def collect_metrics(env:TrainEnv, action, jerks_arr, actions_arr, positions_arr, \
                    speeds_arr, acceleration_arr, punctuality_arr, safety_arr, energy_arr):

    positions_arr.append(env.train.position)
    actions_arr.append(action)
    speeds_arr.append(env.train.speed * 3.6)
    acceleration_arr.append(env.train.acceleration)
    jerks_arr.append(env.max_jerk)
    punctuality_arr.append(env.delta_time_left)
    safety_arr.append(env.speed_limit - env.train.speed)
    energy_arr.append(env.E)


def plot_metrics(env:TrainEnv, plot_path:Path, positions_arr, speeds_arr, \
                 accelerations_arr, actions_arr, punctuality_arr, comfort_arr, \
                 energy_arr):
    
    '''
    This function takes in the metrics saved during evaluation and plots graphs
    to provide a possibility for visual analysis of the evaluation results.

    '''

    fig, axs = plt.subplots(3, 2)
    fig.set_figwidth(15)
    fig.set_figheight(2*4)
    fig.tight_layout()

    # Plot speed limits, estimated speeds and the speed of the train during journey
    axs[0,0].plot(positions_arr, speeds_arr, label='Actual Speed')
    axs[0,0].step([segment[0] for segment in env.journey_speed_limit_segments] + [env.journey_speed_limit_segments[-1][1]],
            [segment[2] * 3.6 for segment in env.journey_speed_limit_segments] + [env.journey_speed_limit_segments[-1][2]],
            where='post', color='red', label='Speed Limit')
    for index, segment in enumerate(env.journey_speed_limit_segments):
        if index == 0:
            axs[0,0].plot([segment[0], segment[1]], [env.estimated_velocity_limit_segments[index] * 3.6, \
                                                     env.estimated_velocity_limit_segments[index] * 3.6], color='green', \
                                                     linestyle='dashed', label='Estimated Speed')
        else:
            axs[0,0].plot([segment[0], segment[1]], [env.estimated_velocity_limit_segments[index] * 3.6, \
                                                     env.estimated_velocity_limit_segments[index] * 3.6], color='green', \
                                                     linestyle='dashed')
    axs[0,0].set_title('Speed Profile')
    axs[0,0].set_xlabel('Position [m]')
    axs[0,0].set_ylabel('Speed [kmph]')
    axs[0,0].set_ylim([0.0, None])
    axs[0,0].set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axs[0,0].grid(color='lightgrey', linestyle='--')
    axs[0,0].legend()

    # Plot the actions that the agent takes over the journey and also plot the acceleration of the train on the same plot
    axs[1,0].plot(positions_arr, actions_arr, label='RTB_req', color='blue')
    axs[1,0].set_title('RTB_req and Acceleration')
    axs[1,0].set_xlabel('Position [m]')
    axs[1,0].set_ylabel('RTB Request [%]', color='blue')
    axs[1,0].set_ylim([-105, 105])
    axs[1,0].set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axs[1,0].tick_params(axis='y', labelcolor='blue')

    axx = axs[1,0].twinx()
    axx.plot(positions_arr, accelerations_arr, label='Acceleration', color='tab:red')
    axx.set_ylabel('Acceleration [m/s^2]', color='tab:red')
    axx.set_ylim([-1.2, 1.2])
    axx.set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axx.tick_params(axis='y', labelcolor='tab:red')
    
    axs[1,0].grid(color='lightgrey', linestyle='--')
    axs[1,0].legend()

    # Plot the slope profile of the track on which the train will run
    gradients_arr = []
    for position in positions_arr:
        for segment in env.track.gradient_segments:
            if segment[0] <= position < segment[1]:
                gradients_arr.append(segment[2])
                break
    axs[2,0].plot(positions_arr, gradients_arr)
    axs[2,0].set_title('Track Data')
    axs[2,0].set_xlabel('Position [m]')
    axs[2,0].set_ylabel('Slope [\u2030]')
    axs[2,0].set_ylim([-25.0, 25.0])
    axs[2,0].set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axs[2,0].grid(color='lightgrey', linestyle='--')

    # Plot jerk experienced by the passengers during journey
    axs[0,1].plot(positions_arr, comfort_arr)
    axs[0,1].plot([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10], [4.0, 4.0], color='red')
    axs[0,1].set_title('Passenger Comfort')
    axs[0,1].set_xlabel('Position [m]')
    axs[0,1].set_ylabel('Jerk [m/s^3]')
    axs[0,1].set_ylim([-0.2, None])
    axs[0,1].set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axs[0,1].grid(color='lightgrey', linestyle='--')

    # Plot the running punctuality error over the journey
    axs[1,1].plot(positions_arr, punctuality_arr)
    axs[1,1].set_title('Punctuality Error')
    axs[1,1].set_xlabel('Position [m]')
    axs[1,1].set_ylabel('Punctuality Error [s]')
    axs[1,1].set_ylim([None, None])
    axs[1,1].set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axs[1,1].grid(color='lightgrey', linestyle='--')

    # Plot the estimated energy that the train consumes over the journey
    axs[2,1].plot(positions_arr, energy_arr)
    axs[2,1].set_title('Energy Estimate')
    axs[2,1].set_xlabel('Position [m]')
    axs[2,1].set_ylabel('Energy')
    axs[2,1].set_ylim([-0.2, None])
    axs[2,1].set_xlim([env.journey.starting_point[0]-10, env.journey.stopping_point[0]+10])
    axs[2,1].grid(color='lightgrey', linestyle='--')

    # Set plot spaces to avoid overlap of individual plots
    plt.subplots_adjust(top=0.962, bottom=0.064, left=0.06, right=0.983, wspace=0.238, hspace=0.43)

    if not plot_path.is_file():
        plt.savefig(plot_path)
        print('Performance plot saved: ' + str(plot_path))
    else:
        print('Performance plot already exists')
    plt.show()


if __name__ == '__main__':

    main()

