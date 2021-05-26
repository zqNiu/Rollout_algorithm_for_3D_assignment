import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB

N=99999
def rollout_3D_assignment(cost_matrix,maximize=False):
    """Solve the 3D assignment using the rollout algorithm according to Bertsekas's paper

    parameters
    ----------
    cost_matrix : 3D-array   j-m-w  job-machine-worker
        The cost matrix of the assignment group.
    maximize : bool (default: False)
        Calculates a maximum weight matching if true.
    Returns
    -------
    assign : 2D-array   
        corresponding indices giving the optimal assignment
        index: job index     col: machine index    value: worker index
    obj: float
        objective function 

    References
    ----------
    1. [Bertsekas's Rollout Algorithm]
    (http://web.mit.edu/dimitrib/www/Rollout_Constrained_Multiagent.pdf). 
    """
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.ndim != 3:
        raise ValueError("expected a matrix (3-D array), got a %r array"
                         % (cost_matrix.shape,))

    if not (np.issubdtype(cost_matrix.dtype, np.number) or
            cost_matrix.dtype == np.dtype(np.bool_)):
        raise ValueError("expected a matrix containing numerical entries, got %s"
                         % (cost_matrix.dtype,))

    #get the initial solution 
    num_job=num_machine=num_worker=cost_matrix.shape[0]
    assign_jm=-np.ones(num_job)
    current_assign_jm=enforced_separation_heuristics(assign_jm,cost_matrix)["assign_jm"]
    current_obj=enforced_separation_heuristics(assign_jm,cost_matrix)["obj"]

    for j in range(num_job):
        current_branch_solutions=np.zeros(num_machine)
        current_branch_assign=dict()
        for m in range(num_machine):
            temp_assign_jm=assign_jm.copy().astype(int)
            if (temp_assign_jm== m).any():
                continue
            temp_assign_jm[j]=m
            current_branch_solutions[m]=enforced_separation_heuristics(temp_assign_jm,cost_matrix)["obj"]
            current_branch_assign[m]=enforced_separation_heuristics(temp_assign_jm,cost_matrix)["assign_jm"]

        min_solution=np.min(current_branch_solutions)
        min_m_index=np.where(current_branch_solutions==min_solution)[0][0]
        if min_solution<=current_obj and min_solution!=0:
            current_obj=min_solution
            assign_jm[j]=min_m_index
            current_assign_jm=current_branch_assign[min_m_index]

        else:
            if not (assign_jm==current_assign_jm[j]).any():
                assign_jm[j]=current_assign_jm[j]

            
    final_assign=enforced_separation_heuristics(assign_jm,cost_matrix)["assign_2D"]
    final_solution=enforced_separation_heuristics(assign_jm,cost_matrix)["obj"]

    print("rollout result:",final_solution,final_assign)




def enforced_separation_heuristics(assign_jm,cost_matrix,maximize=False):
    """
    parameters
    ----------
    assign_jm: 1D-array j-m job-machine
        some j-m pairs are fixed 
    cost_matrix : 3D-array   j-m-w  job-machine-worker
        The cost matrix of the assignment group.
    maximize : bool (default: False)
        Calculates a maximum weight matching if true.
    Returns
    -------
    assign : 2D-array   
        corresponding indices giving the optimal assignment
        index: job index     col: machine index    value: worker index
    obj: float
        objective function 

    get the solution by using enforced separation heuristics 
    the core thought is decoupling the assignment cost
    by first focusing on assigning machines to workers
    then focusing on assigning jobs to machines
    """

    num_job=num_machine=num_worker=cost_matrix.shape[0]
    job_index_2_real_job=dict()
    machine_index_2_real_machine=dict()
    job_index=machine_index=0
    #step1 decouple the cost
    cost_matrix_m_w=-np.ones((num_machine,num_worker))  
    #row: machine   col:worker
    for j in range(num_job):
        for m in range(num_machine):
            for w in range(num_worker):
                if assign_jm[j]>0:
                    #the j-m pair is fixed  c_mw=c_jmw
                    cost_matrix_m_w[m][w]=cost_matrix[j,int(assign_jm[j]),w]
                else:
                    # c_mw=min{j,c_jmw}
                    cost_matrix_m_w[m][w]=np.min(cost_matrix[:,m,w])

    #step2 solve the assignment between machines and workers
    assign_m_w=auction_asy(cost_matrix_m_w,True)["assign"]
    #step3 calculate the cost between jobs and machines
    job_index=0
    for j in range(num_job):
        if (assign_jm[j]<0):
            # not fixed
            job_index_2_real_job[job_index]=j
            job_index+=1
    machine_index=0       
    for m in range(num_machine):
        if not (assign_jm==m).any():
            # not fixed
            machine_index_2_real_machine[machine_index]=m
            machine_index+=1
    assert(job_index==machine_index)
    cost_matrix_j_m=np.zeros((job_index,machine_index))

    for j in range(job_index):
        for m in range(job_index):
            real_job=job_index_2_real_job[j]
            real_machine=machine_index_2_real_machine[m]
            cost_matrix_j_m[j][m]=cost_matrix[real_job,int(real_machine),int(assign_m_w[int(real_machine)])]
    #step4 solve the assignment between jobs and machines
    if not (assign_jm>-1).all():
        assign_j_m=auction_asy(cost_matrix_j_m,True)["assign"]
        for j in range(job_index):
            real_job=job_index_2_real_job[j]
            real_machine=machine_index_2_real_machine[assign_j_m[j]]
            assign_jm[real_job]=real_machine
  
    
    #step5 calculate objective fuction and output 3D assignment result
    assign_2D=-np.ones((num_job,num_machine))
    for j in range(num_job):
        assign_2D[j][int(assign_jm[j])]=assign_m_w[int(assign_jm[j])]

    obj=0
    for j in range(num_job):
        for m in range(num_machine):
            for w in range(num_worker):
                if(assign_2D[j][m]==w):
                    obj+=cost_matrix[j][m][w]

    return {"assign_2D":assign_2D,"obj":obj,"assign_jm":assign_jm}



    

def auction_asy(cost_matrix,minimize=False):
    """
    Python Implemention of Bertsekas's Auction Algorithm by Zhiqiang Niu
    the source code can be found in https://github.com/zqNiu/Auction-Algorithm-python

     arameters
    ----------
    cost_matrix : array
        The cost matrix of the assignment.
    minimize : bool (default: False)
        Calculates a minimize weight matching if true.
    Returns
    -------
    assign : array
        corresponding indices giving the optimal assignment
        index: row index     value: col index. 
    obj: float
        objective function 
    """
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.ndim != 2:
        raise ValueError("expected a matrix (2-D array), got a %r array"
                         % (cost_matrix.shape,))

    if not (np.issubdtype(cost_matrix.dtype, np.number) or
            cost_matrix.dtype == np.dtype(np.bool_)):
        raise ValueError("expected a matrix containing numerical entries, got %s"
                         % (cost_matrix.dtype,))

    if minimize:
        cost_matrix = -cost_matrix
    
    cost_matrix = cost_matrix.astype(np.double)

    # auction algorithm implement 
    
    num_person=num_object=cost_matrix.shape[0]
    assign=np.asarray([-N]*(num_person))
    epsilon=abs(np.min(cost_matrix))+1
    price=np.asarray([np.min(cost_matrix)]*(num_object))
    
    while(abs(epsilon) >1/num_person):
        assign=np.asarray([-N]*(num_person))
        time_start=time.time()
        while ((assign<0).any()):
            #bidding phase
            #Compute the bids of each unassigned individual person and store them in temp array
            bid_value=np.zeros((num_person,num_object))  # row:person col:object
            best_margin=-N   
            best_margin_j_index=-N
            second_margin=-N
            second_margin_j_index=-N
            for i in range(num_person):
                if assign[i]<0:
                    # unassigned
                    # Need calculate the best(max) and second best(max) value of each object to this person
                    for j in range(num_object):
                        margin=cost_matrix[i][j]-price[j]
                        if margin>best_margin:
                            best_margin=margin
                            best_margin_j_index=j
                        elif margin>second_margin:
                            second_margin=margin
                            second_margin_j_index=j

                    if (second_margin==-N):   # only one positive bid for j
                        second_margin=best_margin

                    bid_value[i][best_margin_j_index]=cost_matrix[i][best_margin_j_index]-\
                                                      second_margin+epsilon
                    # also =price[best_margin_j_index]+best_margin-second_margin+epsilon
            #assignment phase
            #Each object which has received a bid determines the highest bidder and 
            #updates its price accordingly
            bid_value_T=np.transpose(bid_value)  # row:object col:person
            for j in range(num_object):
                bid_for_j=bid_value_T[j]
                if((bid_for_j>0).any()):
                    max_bid=np.max(bid_for_j)
                    max_bid_person=np.where(bid_for_j==np.max(bid_for_j))[0][0]
                    if np.where(assign==j)[0].shape[0]>0:
                        # j has been assigned, corresponding i need be set to unassigned 
                        i_index=np.where(assign==j)[0][0]
                        assign[i_index]=-N
                    assign[max_bid_person]=j
                    price[j]=max_bid

        epsilon=epsilon/2
        
    obj=0
    for i in range(num_person):
        obj+=cost_matrix[i][assign[i]]

    if minimize:
        obj=-obj
    return {"assign":assign,"obj":obj,"price":price}

def gurobi_solve(cost_matrix):
    """ Three-Dimensional Assignment problem
           sum(c_jmw*x_jmw)
           s.t.
           sum(j,x_jmw)=e=1  for each (m,w) 
           sum(m,x_jmw)=e=1  for each (j,w)
           sum(w,x_jmw)=e=1  for each (j,m)
    """
    #变量
    model=gp.Model('three_dimensional_assing_problem')
    model.setParam('OutputFlag', 0)
    num_job=num_machine=num_worker=cost_matrix.shape[0]
    x=[[[0 for j in range(num_job)] for m in range(num_machine)] for w in range(num_worker)]
    for j in range(num_job):
        for m in range(num_machine):
            for w in range(num_worker):
                x[j][m][w]=model.addVar(lb=0,ub=1,vtype=GRB.BINARY)
    #目标 
    obj=gp.LinExpr()
    for j in range(num_job):
        for m in range(num_machine):
            for w in range(num_worker):
                obj+= cost_matrix[j][m][w]*x[j][m][w]

    model.setObjective(obj,GRB.MINIMIZE)
    #约束
    for j in range(num_job):
        sum_1=gp.LinExpr()
        for m in range(num_machine):
            for w in range(num_worker):
                sum_1+=x[j][m][w]
           
        model.addConstr(sum_1==1)

    for m in range(num_machine):
        sum_2=gp.LinExpr()
        for j in range(num_job):
            for w in range(num_worker):
                sum_2+=x[j][m][w]
           
        model.addConstr(sum_2==1)

    for w in range(num_worker):
        sum_3=gp.LinExpr()
        for j in range(num_job):
            for m in range(num_machine):
                sum_3+=x[j][m][w]
           
        model.addConstr(sum_3==1)
    #求解    
    model.write('3D_assign.lp')
    model.optimize()

    assign=-np.ones((num_job,num_machine))
    for j in range(num_job):
        for m in range(num_machine):
            for w in range(num_worker):
                if (x[j][m][w].x>0):
                    assign[j][m]=w
                    
    obj=obj.getValue()
    print("gurobi result:",obj,assign)
    return {"assign":assign,"obj":obj}


if __name__=="__main__":
    cost_matrix = np.random.randint(low=0,high=100,size=(5,5,5))
    print(cost_matrix)
    gurobi_solve(cost_matrix)
    rollout_3D_assignment(cost_matrix)
