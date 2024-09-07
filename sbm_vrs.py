import pandas as pd
import numpy as np
from pulp import *

def GetObjCoeff(m, n, l, X0):
    
    obj_coeff = [1]
    zero_n = list(np.repeat(0, n, axis = 0))
    zero_l = list(np.repeat(0, l, axis = 0))
    input_weighted = (-1/m) * (1/X0)
    
    obj_coeff.extend(zero_n)
    obj_coeff.extend(list(input_weighted))
    obj_coeff.extend(zero_l)
            
    return obj_coeff

def DefineVar(obj_coeff):
    
    n_var = len(obj_coeff)
    variables = LpVariable.dicts("V", list(range(n_var)), lowBound=0, cat="Continuous")
        
    return variables

def SetObjCoeff(obj_coeff, variables, problem):
    
    obj_func = 0
    for idx, coeff in enumerate(obj_coeff):
        obj_func += coeff * variables[idx]
        
    problem += obj_func
        
    return problem

def LinearConstraint(m, n, l, Y0, variables):
    
    con_coeff = [1]
    zero_n = list(np.repeat(0, n, axis = 0))
    zero_m = list(np.repeat(0, m, axis = 0))
    output_weighted = [(1/l) * (1/Y0)]
    
    con_coeff.extend(zero_n)
    con_coeff.extend(zero_m)
    con_coeff.extend(output_weighted)
    
    constraint = 0
    for idx, coeff in enumerate(con_coeff):
        constraint += coeff * variables[idx]

    return constraint

def InputConstraint(X0, front_X, m, l, variables):

    neg_X0 = -X0
    input_const_list = []
    
    if m == 1:
        diag_mat = [elem for elem_list in -np.identity(m) for elem in list(elem_list)]
        zero_mat = [elem for elem_list in np.zeros((m,l)) for elem in list(elem_list)]
        const_list = []
        const_list.append(float(neg_X0))
        const_list.extend(list(front_X.iloc[0,:]))
        const_list.extend(diag_mat)
        const_list.extend(zero_mat)
        
        constraint = 0
        for idx, coeff in enumerate(const_list):
            constraint += coeff * variables[idx]    
        input_const_list.append(constraint)
        
    else :
        diag_mat = np.identity(m)
        zero_mat = np.zeros((m,l))

        const_mat = pd.concat([neg_X0.reset_index(drop=True),
                               front_X.reset_index(drop=True),
                               pd.DataFrame(diag_mat).reset_index(drop=True), 
                               pd.DataFrame(zero_mat).reset_index(drop=True)], axis=1).to_numpy()

        for each_input in const_mat:
            constraint = 0
            for idx, coeff in enumerate(each_input):
                constraint += coeff * variables[idx]

            input_const_list.append(constraint)

    return input_const_list

def OutputConstraint(Y0, front_Y, m, l, variables):
        
    neg_Y0 = -Y0
    output_const_list = []
    
    if l == 1:
        diag_mat = [elem for elem_list in -np.identity(l) for elem in list(elem_list)]
        zero_mat = [elem for elem_list in np.zeros((l,m)) for elem in list(elem_list)]
        const_list = []
        const_list.append(float(neg_Y0))
        const_list.extend(list(front_Y.iloc[0,:]))
        const_list.extend(zero_mat)
        const_list.extend(diag_mat)
        
        constraint = 0
        for idx, coeff in enumerate(const_list):
            constraint += coeff * variables[idx]    
        output_const_list.append(constraint)
        
    else :
        diag_mat = np.identity(l)
        zero_mat = zero_mat = np.zeros((l,m))
        const_mat = pd.concat([neg_Y0.reset_index(drop=True),
                               front_Y.reset_index(drop=True),
                               pd.DataFrame(zero_mat).reset_index(drop=True), 
                               pd.DataFrame(diag_mat).reset_index(drop=True)], axis=1).to_numpy() 
        
        for each_output in const_mat:            
            constraint = 0
            for idx, coeff in enumerate(each_output):
                constraint += coeff * variables[idx]
                
            output_const_list.append(constraint)
    
    return output_const_list

def VrsConstraint(n, m, l, variables):
    
    lambda_coeff = [-1]
    lambda_coeff_n = list(np.repeat(1, n, axis = 0))
    zero_ms = list(np.repeat(0, m+l, axis = 0))
    
    lambda_coeff.extend(lambda_coeff_n)
    lambda_coeff.extend(zero_ms)
    
    lambda_const = 0
    for idx, coeff in enumerate(lambda_coeff):
        lambda_const += coeff * variables[idx]
        
    return lambda_const


def AddConstraints(problem, X0, Y0, m, n, l, front_X, front_Y, variables):
    
    linear_coeff = LinearConstraint(m, n, l, Y0, variables)
    problem += linear_coeff == 1
    
    input_const = InputConstraint(X0, front_X, m, l, variables)
    for each_input_const in input_const:
        problem += each_input_const == 0
    
    output_const = OutputConstraint(Y0, front_Y, m, l, variables)
    for each_output_const in output_const:
        problem += each_output_const == 0
    
    vrs_const = VrsConstraint(n, m, l, variables)
    problem += vrs_const == 0
    
    return problem

def Labeling(n, m, l):
    
    label_list = ["dmu", "sbm.vrs_eff"]
    
    for idx in list(range(1,n+1)):
        lambda_label = "lambda_" + str(idx)
        label_list.append(lambda_label)
        
    for input_idx in list(range(1, m+1)):
        slack_x_label = "x_slack_" + str(input_idx)
        label_list.append(slack_x_label)
        
    for output_idx in list(range(1, l+1)):
        slack_y_label = "y_slack_" + str(output_idx)
        label_list.append(slack_y_label)
        
    return label_list

def BenchMarking(n, dmu, df):
    
    benchmark_list = []
    for row_idx in list(range(0, n)):
        benchmarks = []
        for col_idx, each_lambda in enumerate(list(df.iloc[row_idx,2:n+2])):
            if each_lambda == 0:
                continue
            benchmark = dmu[col_idx]

            if benchmark == dmu[row_idx]:
                continue

            benchmarks.append(benchmark)
        benchmark_str = ','.join(benchmarks)
        benchmark_list.append(benchmark_str)
        
    df["benchmark"] = benchmark_list
    
    return df


def SolveProblem(input_file_name, output_file_name, n_output=1):

    raw_data = pd.read_csv(input_file_name).iloc[:,1:]
    dmu = pd.read_csv(input_file_name).iloc[:,0]
    l = n_output
    m = len(raw_data.columns) - l
    n = len(raw_data)
    front_Y = raw_data.transpose().iloc[:l,:]
    front_X = raw_data.transpose().iloc[l:,:]

    result = []
    for idx in list(range(len(raw_data))):
        X0 = front_X.iloc[:,idx]
        Y0 = front_Y.iloc[:,idx]

        problem = LpProblem("Slack-based Model", LpMinimize)
        obj_coeff = GetObjCoeff(m, n, l, X0)
        variables = DefineVar(obj_coeff)
        problem = SetObjCoeff(obj_coeff, variables, problem)
        problem = AddConstraints(problem, X0, Y0, m, n, l, front_X, front_Y, variables)
        
        problem.solve(GLPK(msg = 0))
        eff = pulp.value(problem.objective)
        
        solution_list = []
        # Do not use 'problem.variable()' ; the order of var is awry  
        for idx, variable in variables.items():

            if idx == 0:
                continue 

            each_solution = variable.varValue/variables[0].varValue
            solution_list.append(each_solution)
    
        each_result = [eff]
        each_result.extend(solution_list)
        result.append(each_result)
    
    label_list = Labeling(n,m,l)
    df = pd.concat([dmu.reset_index(drop=True),
                    pd.DataFrame(result).reset_index(drop=True)], axis=1)
    df.columns = label_list
        
    df = BenchMarking(n, dmu, df)
    df.to_csv(output_file_name, index=False)

if __name__ == '__main__':
    SolveProblem(input_file_name, output_file_name, n_output = 1)