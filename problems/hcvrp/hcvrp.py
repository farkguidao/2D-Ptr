import torch

class HcvrpEnv:
    def __init__(self,input,scale=(1,40,1)):
        '''
        :param input:
        input:{
            'loc':  batch_size, graph_size, 2
            'demand': batch_size, graph_size
            'depot': batch_size, 2
            'capacity': batch_size, vehicle_num
            'speed': batch_size, vehicle_num
        }
        :param scale: used to output normalized state (coords,demand,speed)
        '''
        self.device = input['loc'].device
        self.batch_size = input['loc'].shape[0]
        self.bs_index = torch.arange(self.batch_size,device = self.device)
        self.step = 0
        self.scale_coords,self.scale_demand,self.scale_speed = scale
        self.initial_node_state(input['loc'],input['demand'],input['depot'])
        self.initial_veh_state(input['capacity'], input['speed'])
    def initial_node_state(self,loc,demand,depot):
        '''
        :param loc:  customer coordinates [batch_size, graph_size,2]
        :param demand: customer demands [batch_size, graph_size]
        :param depot: depot coordinates [batch_size, 2]
        :return:
        '''
        assert loc.shape[:2] == demand.shape, "The custumer's loc and demand shape do not match"
        self.customer_num = loc.shape[1]
        self.N = loc.shape[1]+1 # Let N represent the graph size
        self.coords = torch.cat([depot.unsqueeze(1),
                                 loc],dim=1) # batch_size, N, 2
        self.demand = torch.cat([torch.zeros_like(demand[:,[0]]),
                                 demand],dim=1) # batch_size, N
        self.visited = torch.zeros_like(self.demand).bool() # batch_size, N
        self.visited[:,0] = True # start from depot, so depot is visited
    def all_finished(self):
        '''
        :return: Are all tasks finished?
        '''
        return self.visited.all()

    def finished(self):
        '''
        :return: [bs],true or false, is each task finished?
        '''
        return self.visited.all(-1)

    def get_all_node_state(self):
        '''
        :return: [bs,N+1,3], get node initial features
        '''
        return torch.cat([self.coords/self.scale_coords,
                          self.demand.unsqueeze(-1)/self.scale_demand],dim = -1) # batch_size, N, 3

    def initial_veh_state(self,capacity,speed):
        '''
        :param capacity:  batch_size, veh_num
        :param speed: batch_size, veh_num
        :return
        '''
        assert capacity.size() == speed.size(), "The vehicle's speed and capacity shape do not match"
        self.veh_capacity = capacity
        self.veh_speed = speed
        self.veh_num = capacity.shape[1]
        self.veh_time = torch.zeros_like(capacity)  # batch_size, veh_num
        self.veh_cur_node = torch.zeros_like(capacity).long() # batch_size, veh_num
        self.veh_used_capacity = torch.zeros_like(capacity)
        # a util vector
        self.veh_index = torch.arange(self.veh_num, device=self.device)

    def min_max_norm(self,data):
        '''
        deprecated
        :param data:
        :return:
        '''
        # bs，M
        min_data = data.min(-1,keepdim=True)[0]
        max_data = data.max(-1, keepdim=True)[0]
        return (data-min_data)/(max_data-min_data)
    def get_all_veh_state(self):
        '''
        :return: [bs,M,4]
        # time，capacity，usage capacity，speed
        '''

        veh_cur_coords = self.coords[self.bs_index.unsqueeze(-1),
                                     self.veh_cur_node] # batch_size, veh_num, 2

        return torch.cat([
            self.veh_time.unsqueeze(-1),
                          self.veh_capacity.unsqueeze(-1)/self.scale_demand,
                          self.veh_used_capacity.unsqueeze(-1)/self.scale_demand,
                          self.veh_speed.unsqueeze(-1)/self.scale_speed,
                          # veh_cur_coords/self.scale_coords
        ],dim=-1)

    def get_veh_state(self,veh):
        # deprecated
        '''
        :param veh: veh_index，batch_size
        :return:
        '''
        all_veh_state = self.get_all_veh_state() # bs,veh_num,4
        return all_veh_state[self.bs_index,veh] # bs,4


    def action_is_legal(self,veh,next_node):
        # deprecated
        return self.demand[self.bs_index, next_node] <= (self.veh_capacity - self.veh_used_capacity)[self.bs_index, veh]

    def update(self, veh, next_node):
        '''
        input action tuple and update the env
        :param veh: [batch_size,]
        :param next_node: [batch_size,]
        :return:
        '''
        # select node must be unvisited,except depot
        assert not self.visited[self.bs_index,next_node][next_node!=0].any(),"Wrong solution: node has been selected !"
        # Note that demand<=remaining_capacity==capacity-usage_capacity
        assert (self.demand[self.bs_index,next_node] <=
                (self.veh_capacity-self.veh_used_capacity)[self.bs_index,veh]).all(),"Wrong solution: the remaining capacity of the vehicle cannot satisfy the node !"

        # update vehicle time，
        last_node = self.veh_cur_node[self.bs_index,veh]
        old_coords,new_coords = self.coords[self.bs_index,last_node],self.coords[self.bs_index,next_node]
        length = torch.norm(new_coords-old_coords,p=2,dim=1)
        time_add = length / self.veh_speed[self.bs_index,veh]
        self.veh_time[self.bs_index,veh] += time_add

        # update the used_capacity
        new_veh_used_capacity = self.veh_used_capacity[self.bs_index, veh] + self.demand[self.bs_index,next_node]
        new_veh_used_capacity[next_node==0] = 0 # 回到仓库后装满车辆
        self.veh_used_capacity[self.bs_index, veh] = new_veh_used_capacity

        # update the node index where the vehicle stands
        self.veh_cur_node[self.bs_index,veh] = next_node
        self.step += 1
        # print(self.step)

        # update visited vector
        self.visited[self.bs_index,next_node]=True

    def all_go_depot(self):
        '''
        All vehicle go back the depot
        :return:
        '''
        veh_list = torch.arange(self.veh_num,device = self.device)
        depot = torch.zeros_like(self.bs_index)
        for i in veh_list:
            self.update(i.expand(self.batch_size),depot)

    def get_cost(self,obj):
        self.all_go_depot()
        if obj=='min-max':
            return self.veh_time.max(-1)[0]
        elif obj=='min-sum':
            return self.veh_time.sum(-1)
    def get_action_mask(self):
        # cannot select a visited node except the depot
        visited_mask = self.visited.clone() # bs,N+1
        visited_mask[:,0]=False
        # Here, clone() is important for avoiding the bug from expand()
        visited_mask = visited_mask.unsqueeze(1).expand(self.batch_size, self.veh_num, self.N).clone() # bs,M,N+1
        # Vehicle cannot stay in place to avoid visiting the depot twice,
        # otherwise an infinite loop will easily occur
        visited_mask[self.bs_index.unsqueeze(-1),self.veh_index.unsqueeze(0),self.veh_cur_node]=True
        # capacity constraints
        demand_mask = (self.veh_capacity - self.veh_used_capacity).unsqueeze(-1) < self.demand.unsqueeze(1) # bs,M,N+1
        mask = visited_mask | demand_mask
        # Special setting for batch processing,
        # because the finished task will have a full mask and raise an error
        mask[self.finished(),0,0]=False
        return mask

    @staticmethod
    def caculate_cost(input,solution,obj):
        '''
        :param input: equal to __init__
        :param solution: (veh,next_node): [total_step, batch_size],[total_step, batch_size]
        :param obj: 'min-max' or 'min-sum'
        :return: cost : batch_size
        '''

        env = HcvrpEnv(input)
        for veh,next_node in zip(*solution):
            env.update(veh,next_node)
        return env.get_cost(obj)