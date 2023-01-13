import numpy as np
import torch.nn.functional as F
import torch

class ReplayBuffer_list():
    """
    경험 재현 버퍼 list 버전
    """
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        #self.vi_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        
        #self.ds_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)

        self.max_size = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def put(self, sample):
        s, a, r, p, vi = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        #self.vi_mem[self._idx] = vi
        #self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        s_lst = torch.tensor(np.vstack(self.ss_mem[idxs]),dtype=torch.float32)
        a_lst = torch.tensor(np.vstack(self.as_mem[idxs]),dtype=torch.int64)
        r_lst = torch.tensor(np.vstack(self.rs_mem[idxs]),dtype=torch.int64)    
        s_prime_lst = torch.tensor(np.vstack(self.ps_mem[idxs]),dtype=torch.float32)
        #vi_lst = torch.tensor(np.vstack(self.vi_mem[idxs]),dtype=torch.int32)

        experiences = s_lst, \
                      a_lst, \
                      r_lst, \
                      s_prime_lst, \
                  
                      
        return experiences

    def __len__(self):
        return self.size

class ReplayBuffer_vi():
    """
    경험 재현 버퍼 list 버전
    """
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.vi_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        
        #self.ds_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)

        self.max_size = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def put(self, sample):
        s, a, r, p, vi = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.vi_mem[self._idx] = vi
        #self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        s_lst = torch.tensor(np.vstack(self.ss_mem[idxs]),dtype=torch.float32)
        a_lst = torch.tensor(np.vstack(self.as_mem[idxs]),dtype=torch.int64)
        r_lst = torch.tensor(np.vstack(self.rs_mem[idxs]),dtype=torch.int64)    
        s_prime_lst = torch.tensor(np.vstack(self.ps_mem[idxs]),dtype=torch.float32)
        vi_lst = torch.tensor(np.vstack(self.vi_mem[idxs]),dtype=torch.int32)

        experiences = s_lst, \
                      a_lst, \
                      r_lst, \
                      s_prime_lst, \
                      vi_lst, \
                      
        return experiences

    def __len__(self):
        return self.size

class ReplayBuffer_vi_day():
    """
    경험 재현 버퍼 list 버전
    """
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.vi_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.day_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)

        self.max_size = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def put(self, sample):
        s, a, r, p, vi, day = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.vi_mem[self._idx] = vi
        self.day_mem[self._idx] = day

        
        #self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        s_lst = torch.tensor(np.vstack(self.ss_mem[idxs]),dtype=torch.float32)
        a_lst = torch.tensor(np.vstack(self.as_mem[idxs]),dtype=torch.int64)
        r_lst = torch.tensor(np.vstack(self.rs_mem[idxs]),dtype=torch.int64)    
        s_prime_lst = torch.tensor(np.vstack(self.ps_mem[idxs]),dtype=torch.float32)
        vi_lst = torch.tensor(np.vstack(self.vi_mem[idxs]),dtype=torch.int32)
        day_lst = torch.tensor(np.vstack(self.day_mem[idxs]),dtype=torch.int32)
        
        experiences = s_lst, \
                      a_lst, \
                      r_lst, \
                      s_prime_lst, \
                      vi_lst,\
                      day_lst, \
          
        return experiences

    def __len__(self):
        return self.size

class ReplayBuffer_vi_market():
    """
    경험 재현 버퍼 list 버전
    """
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.vi_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.day_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.time_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)

        self.max_size = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def put(self, sample):
        s, a, r, p, vi, day ,time = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.vi_mem[self._idx] = vi
        self.day_mem[self._idx] = day
        self.time_mem[self._idx] = time

        
        #self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        s_lst = torch.tensor(np.vstack(self.ss_mem[idxs]),dtype=torch.float32)
        a_lst = torch.tensor(np.vstack(self.as_mem[idxs]),dtype=torch.int64)
        r_lst = torch.tensor(np.vstack(self.rs_mem[idxs]),dtype=torch.int64)    
        s_prime_lst = torch.tensor(np.vstack(self.ps_mem[idxs]),dtype=torch.float32)
        vi_lst = torch.tensor(np.vstack(self.vi_mem[idxs]),dtype=torch.int32)
        day_lst = torch.tensor(np.vstack(self.day_mem[idxs]),dtype=torch.int32)
        time_lst = torch.tensor(np.vstack(self.time_mem[idxs]),dtype=torch.int32)
        
        experiences = s_lst, \
                      a_lst, \
                      r_lst, \
                      s_prime_lst, \
                      vi_lst,\
                      day_lst, \
                      time_lst, \
          
        return experiences

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer():
    """
    경험 재현 버퍼 우선순위 방식
    """
    def __init__(self, 
                 buffer_limit=15000, 
                 batch_size=64, 
                 rank_based=False,
                 alpha=0.6, 
                 beta0=0.1, 
                 beta_rate=0.99992):
        self.buffer_limit = buffer_limit
        self.memory = np.empty(shape=(self.buffer_limit, 2), dtype=np.ndarray)
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based 
        self.alpha = alpha
        self.beta = beta0 
        self.beta0 = beta0 
        self.beta_rate = beta_rate

    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def put(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[
                :self.n_entries, 
                self.td_error_index].max()
        self.memory[self.next_index, 
                    self.td_error_index] = priority
        self.memory[self.next_index, 
                    self.sample_index] = np.array(sample)
        self.n_entries = min(self.n_entries + 1, self.buffer_limit)
        self.next_index += 1
        self.next_index = self.next_index % self.buffer_limit

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        return self.beta

    def sample(self, batch_size=None):
        # beta 조절 후 0으로 채워진 행 삭제
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        # rank 기반 순위 결정
        if self.rank_based:
            priorities = 1/(np.arange(self.n_entries) + 1)
        else: # proportional
            priorities = entries[:, self.td_error_index] + 1e-6
        scaled_priorities = priorities**self.alpha       

        # 순위 확률화 
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64)

        # 중요도 및 정규화 계산
        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        
        # 샘플링
        samples = np.array([entries[idx] for idx in idxs])
        
        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = torch.tensor(np.vstack(normalized_weights[idxs]))

        samples_stacks[0] = torch.tensor(samples_stacks[0],dtype=torch.float32)
        samples_stacks[1] = torch.tensor(samples_stacks[1], dtype=torch.int64)
        samples_stacks[2] = torch.tensor(samples_stacks[2], dtype=torch.int64)    
        samples_stacks[3] = torch.tensor(samples_stacks[3], dtype=torch.float32)
        samples_stacks[4] = torch.tensor(samples_stacks[4], dtype=torch.int32)
        samples_stacks[5] = torch.tensor(samples_stacks[5], dtype=torch.int32)

        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return str(self.memory[:self.n_entries])
    
    def __str__(self):
        return str(self.memory[:self.n_entries])