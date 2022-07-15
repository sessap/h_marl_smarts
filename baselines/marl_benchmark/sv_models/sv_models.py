import numpy as np
import torch
import gpytorch
import pickle
import matplotlib.pyplot as plt

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=[2,4,5]))+gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=[2,6,7]))

    def forward(self, sv_state):
        mean_x = self.mean_module(sv_state)
        covar_x = self.covar_module(sv_state)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel_2(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel_2, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module =  gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=[2,3]))

    def forward(self, sv_state):
        mean_x = self.mean_module(sv_state)
        covar_x = self.covar_module(sv_state)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SV_GPModel:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        models = []
        likelihoods = []
        if len(train_y.shape)>1:
            for i in range(train_y.shape[1]):
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihoods.append(likelihood)
                if i == 0:
                    models.append(ExactGPModel_2(train_x, train_y[:,i], likelihood))
                else:
                    models.append(ExactGPModel(train_x, train_y[:,i], likelihood))
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            models.append(ExactGPModel(train_x, train_y, likelihood))
        self.model = gpytorch.models.IndependentModelList(*models).double()
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihoods)

        self.sv_id = None

    def train(self):
        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)
        training_iter = 50

        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},  # Includes all submodels and all likelihood parameters
        ], lr=0.1)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            loss = -mll(output, self.model.train_targets)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
            optimizer.step()

    def predict(self, state):
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_preds = self.likelihood(*self.model(*[state for i in range(self.model.num_outputs)]))
        return observed_preds

    def save_model(self, path='model_state.pth'):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path='model_state.pth'):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def add_new_data(self, new_inputs, new_targets):
        inputs = torch.cat([self.train_x, new_inputs])
        targets = torch.cat([self.train_y, new_targets])
        if len(targets.shape)>1:
            for i, m in enumerate(self.model.models):
                m.set_train_data(inputs, targets[:,i], strict=False)
        else:
            self.model = self.model.set_train_data(inputs, targets, strict=False)
        self.train()

    @staticmethod
    def compute_sv_ego_states(env_obs):
        sv_ego_states = {}
        for agent_id in env_obs:
            for neigh_state in env_obs[agent_id].neighborhood_vehicle_states:
                if neigh_state.id[0:5] != 'AGENT' and neigh_state.id not in sv_ego_states.keys():
                    sv_ego_states[neigh_state.id] = neigh_state
        return sv_ego_states


    @staticmethod
    def compute_sv_state_v2(sv_id, env_obs):
        sv_state = None
        sv_ego_states = SV_GPModel.compute_sv_ego_states(env_obs)
        if sv_id in sv_ego_states.keys():
            sv_state = [sv_ego_states[sv_id].position[0],
                        sv_ego_states[sv_id].position[1],
                        sv_ego_states[sv_id].speed,
                        float(sv_ego_states[sv_id].heading)]
            dist_front = -100
            dist_back = 100
            for agent_id in env_obs:
                agent_pos = [env_obs[agent_id].ego_vehicle_state.position[0],
                             env_obs[agent_id].ego_vehicle_state.position[1]]
                if abs(agent_pos[1]-sv_ego_states[sv_id].position[1])<1.7:
                    dist_x = sv_ego_states[sv_id].position[0] - agent_pos[0]
                    if dist_x > 0 and dist_x < dist_back:
                        dist_back = dist_x
                    elif dist_x < 0 and abs(dist_x) < abs(dist_front):
                        dist_front = dist_x
            sv_state.extend([dist_front/10, dist_back/10])
        return sv_state

    @staticmethod
    def compute_sv_state(sv_id, env_obs):
        """ State contains :
                - position x
                - position y
                - speed
                - orientation
                - frontal distance to closest vehicle1, 50 if not present
                - bakward distance to closest vehicle1, 50 if not present
                - frontal distance to closest vehicle2, 50 if not present
                - bakward distance to closest vehicle2, 50 if not present
        """
        sv_state = None
        sv_ego_states = SV_GPModel.compute_sv_ego_states(env_obs)
        if sv_id in sv_ego_states.keys():
            sv_state = [sv_ego_states[sv_id].position[0],
                        sv_ego_states[sv_id].position[1],
                        sv_ego_states[sv_id].speed,
                        float(sv_ego_states[sv_id].heading)]
            for agent_id in ['AGENT-0', 'AGENT-1']:
                if agent_id in env_obs:
                    agent_pos = [env_obs[agent_id].ego_vehicle_state.position[0],
                                 env_obs[agent_id].ego_vehicle_state.position[1]]
                    sv_state.extend([(sv_ego_states[sv_id].position[i] - agent_pos[i]) for i in range(2)])
                else:
                    sv_state.extend([50 for i in range(2)])
        return sv_state

    @staticmethod
    def compute_sv_target(sv_id, env_target_obs, env_obs):
        """Target consists of:
                - traveled distance
                - change in speed
        """
        sv_target = None
        sv_ego_target_states = SV_GPModel.compute_sv_ego_states(env_target_obs)
        sv_ego_states = SV_GPModel.compute_sv_ego_states(env_obs)
        if sv_id in sv_ego_states.keys():
            sv_target = [np.linalg.norm(sv_ego_target_states[sv_id].position - sv_ego_states[sv_id].position),
                         sv_ego_target_states[sv_id].speed - sv_ego_states[sv_id].speed]
        return sv_target

    def predict_state(self, state, etas=None, thompson_sample=False):
        state_t = state.numpy().squeeze()
        assert len(state_t)==8, "expected 8-d state, instead of "+str(len(state_t))
        preds = self.predict(state)
        next_state = {}
        with torch.no_grad():
            delta_space = preds[0].mean.numpy()
            delta_speed = preds[1].mean.numpy()
            if len(preds)>2:
                delta_head = preds[2].mean.numpy()
            else:
                delta_head = 0
            if etas is not None and thompson_sample is False:
                delta_space = delta_space + preds[0].stddev.numpy()*etas[0]*0 # no std. is used to predict next position (GP is usually very accurate)
                delta_speed = delta_speed + preds[1].stddev.numpy()*etas[1]
            if thompson_sample:
                delta_speed = np.random.normal(preds[1].mean.numpy(), preds[1].stddev.numpy())
            next_state['position'] = np.concatenate((state_t[0] - np.sign(state_t[3])*delta_space,
                                                    state_t[1]  + 0*np.cos(state_t[3])*delta_space, # assumes horizontal heading
                                                    np.array([0.01])))
            next_state['speed'] = np.float(state_t[2] + delta_speed)
            next_state['heading'] = np.float( state_t[3] + delta_head)
            next_state['orientation'] = np.array([0, 0, 0, 0])
        return next_state

def data_from_transitions(saved_transitions_pickle, many_episodes=False):
    with open(saved_transitions_pickle, "rb") as f:
        saved_transitions = pickle.load(f)
    if many_episodes is False:
        episode_list = [np.random.randint(0, len(saved_transitions)-1)]
    else:
        episode_list = range(len(saved_transitions))
    sv_states = {}
    sv_targets = {}
    for episode in episode_list:
        for t in range(len(saved_transitions[episode]["actions"]) - 1):
            env_obs = saved_transitions[episode]["env_obs"][t]
            sv_ids = SV_GPModel.compute_sv_ego_states(env_obs).keys()
            for sv_id in sv_ids:
                if sv_id not in sv_states.keys():
                    sv_states[sv_id] = []
                    sv_targets[sv_id] = []
                next_env_obs = saved_transitions[episode]["env_obs"][t + 1]
                if sv_id in SV_GPModel.compute_sv_ego_states(next_env_obs).keys():
                    state_t = SV_GPModel.compute_sv_state(sv_id, env_obs)
                    target_t = SV_GPModel.compute_sv_target(sv_id, next_env_obs, env_obs)
                    sv_states[sv_id].append(state_t)
                    sv_targets[sv_id].append(target_t)

    sv_states_allcars = sum(list(sv_states.values()), [])
    sv_targets_allcars = sum(list(sv_targets.values()), [])
    data_x = torch.from_numpy(np.array(sv_states_allcars))
    data_y = torch.from_numpy(np.array(sv_targets_allcars))

    data_y[0] = data_y[0] + 0.00001*torch.randn(data_y[0].shape)
    data_y[1] = data_y[1] + 0.00001*torch.randn(data_y[1].shape)
    return data_x, data_y

if __name__=='__main__':
    re_train = 1
    if re_train:
        data_x, data_y = data_from_transitions('./sv_models/saved_transitions.pkl', many_episodes=True)

        idx_random =  np.random.choice(np.arange(len(data_x)), size=(1,2)).squeeze()
        idx_compl = list(set(range(len(data_x))).difference(list(idx_random)))
        train_x = data_x[idx_random]
        train_y = data_y[idx_random]
        test_x = data_x[idx_compl]
        test_y = data_y[idx_compl]

        ## Initialize, train, and save model
        SV_model = SV_GPModel(train_x.double(), train_y.double())
        SV_model.train()

        for dir in ['sv_models/sv_init_model/']:
            SV_model.save_model(dir + 'model_state.pth')
            torch.save([train_x, train_y], dir + 'training_data.pth')
        del SV_model, train_x, train_y

    ## Load and Evaluate model
    model_path = 'sv_models/sv_init_model'
    train_x, train_y = torch.load(model_path+'/training_data.pth')
    SV_model = SV_GPModel(train_x, train_y)
    #SV_model.train()
    SV_model.load_model(model_path+'/model_state.pth')

    test_x , test_y = train_x, train_y
    test_x , test_y =  data_from_transitions('sv_models/saved_transitions.pkl', many_episodes=True)

    idx_random = np.arange(0,900)
    test_x_sample = test_x[idx_random]
    test_y_sample = test_y[idx_random]

    observed_preds = SV_model.predict(test_x_sample)

    with torch.no_grad():
        for i, pred in enumerate(observed_preds):
            f, ax = plt.subplots(1, 1, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = pred.confidence_region()
            # Plot training data as black stars
            #ax.plot(test_x[:,i].numpy(), 'r*')
            ax.plot(test_y_sample[:,i].numpy(), 'k*')
            # Plot predictive means as blue line
            ax.plot(pred.mean.numpy(), 'b*')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(range(len(test_y_sample)), lower.numpy(), upper.numpy(), alpha=0.5)
            ax.legend(['Observed Data', 'Mean', 'Confidence'])
            plt.show()