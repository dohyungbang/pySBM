import pandas as pd
import numpy as np
from pulp import *

class SBM_VRS:
    def __init__(self, input_file_name, output_file_name, n_output):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.n_output = n_output
        self.raw_data = pd.read_csv(input_file_name).iloc[:, 1:]
        self.dmu = pd.read_csv(input_file_name).iloc[:, 0]
        self.l = n_output
        self.m = len(self.raw_data.columns) - self.l
        self.n = len(self.raw_data)
        self.front_Y = self.raw_data.transpose().iloc[:self.l, :]
        self.front_X = self.raw_data.transpose().iloc[self.l:, :]

    def GetObjCoeff(m, n, l, X0):
        obj_coeff = [1]
        zero_n = list(np.repeat(0, n, axis=0))
        zero_l = list(np.repeat(0, l, axis=0))
        input_weighted = (-1 / m) * (1 / X0)
        obj_coeff.extend(zero_n)
        obj_coeff.extend(list(input_weighted))
        obj_coeff.extend(zero_l)
        return obj_coeff

    def DefineVar(obj_coeff):
        n_var = len(obj_coeff)
        return LpVariable.dicts("V", list(range(n_var)), lowBound=0, cat="Continuous")

    def SetObjCoeff(obj_coeff, variables, problem):
        obj_func = sum(coeff * variables[idx] for idx, coeff in enumerate(obj_coeff))
        problem += obj_func
        return problem

    def LinearConstraint(m, n, l, Y0, variables):
        con_coeff = [1]
        zero_n = list(np.repeat(0, n, axis=0))
        zero_m = list(np.repeat(0, m, axis=0))
        output_weighted = [(1 / l) * (1 / Y0)]
        con_coeff.extend(zero_n)
        con_coeff.extend(zero_m)
        con_coeff.extend(output_weighted)
        constraint = sum(coeff * variables[idx] for idx, coeff in enumerate(con_coeff))
        return constraint

    def InputConstraint(X0, front_X, m, l, variables):
        neg_X0 = -X0
        input_const_list = []
        diag_mat = np.identity(m)
        zero_mat = np.zeros((m, l))
        const_mat = pd.concat([
            neg_X0.reset_index(drop=True),
            front_X.reset_index(drop=True),
            pd.DataFrame(diag_mat).reset_index(drop=True),
            pd.DataFrame(zero_mat).reset_index(drop=True)
        ], axis=1).to_numpy()

        for each_input in const_mat:
            constraint = sum(coeff * variables[idx] for idx, coeff in enumerate(each_input))
            input_const_list.append(constraint)

        return input_const_list

    def OutputConstraint(Y0, front_Y, m, l, variables):
        neg_Y0 = -Y0
        output_const_list = []
        diag_mat = np.identity(l)
        zero_mat = np.zeros((l, m))
        const_mat = pd.concat([
            neg_Y0.reset_index(drop=True),
            front_Y.reset_index(drop=True),
            pd.DataFrame(zero_mat).reset_index(drop=True),
            pd.DataFrame(diag_mat).reset_index(drop=True)
        ], axis=1).to_numpy()

        for each_output in const_mat:
            constraint = sum(coeff * variables[idx] for idx, coeff in enumerate(each_output))
            output_const_list.append(constraint)

        return output_const_list

    def VrsConstraint(n, m, l, variables):
        lambda_coeff = [-1] + list(np.repeat(1, n)) + list(np.repeat(0, m + l))
        lambda_const = sum(coeff * variables[idx] for idx, coeff in enumerate(lambda_coeff))
        return lambda_const

    def AddConstraints(self, X0, Y0, m, n, l, front_X, front_Y, variables, problem):
        linear_coeff = self.LinearConstraint(m, n, l, Y0, variables)
        problem += linear_coeff == 1

        input_const = self.InputConstraint(X0, front_X, m, l, variables)
        for each_input_const in input_const:
            problem += each_input_const == 0

        output_const = self.OutputConstraint(Y0, front_Y, m, l, variables)
        for each_output_const in output_const:
            problem += each_output_const == 0

        vrs_const = self.VrsConstraint(n, m, l, variables)
        problem += vrs_const == 0

        return problem

    def Labeling(n, m, l):
        label_list = ["dmu", "sbm.vrs_eff"]
        label_list += [f"lambda_{idx + 1}" for idx in range(n)]
        label_list += [f"x_slack_{idx + 1}" for idx in range(m)]
        label_list += [f"y_slack_{idx + 1}" for idx in range(l)]
        return label_list

    def BenchMarking(n, dmu, df):
        benchmark_list = []
        for row_idx in range(n):
            benchmarks = [
                dmu[col_idx]
                for col_idx, each_lambda in enumerate(df.iloc[row_idx, 2:n + 2])
                if each_lambda > 0 and dmu[col_idx] != dmu[row_idx]
            ]
            benchmark_list.append(",".join(benchmarks))
        df["benchmark"] = benchmark_list
        return df

    def Solve(self):
        result = []
        for idx in range(len(self.raw_data)):
            X0 = self.front_X.iloc[:, idx]
            Y0 = self.front_Y.iloc[:, idx]
            problem = LpProblem("Slack-based Model", LpMinimize)
            obj_coeff = self.GetObjCoeff(self.m, self.n, self.l, X0)
            variables = self.DefineVar(obj_coeff)
            problem = self.SetObjCoeff(obj_coeff, variables, problem)
            problem = self.AddConstraints(X0, Y0, self.m, self.n, self.l, self.front_X, self.front_Y, variables, problem)

            problem.solve(GLPK(msg=0))
            eff = pulp.value(problem.objective)
            solution_list = [
                variable.varValue / variables[0].varValue
                for idx, variable in variables.items()
                if idx != 0
            ]
            result.append([eff] + solution_list)

        label_list = self.Labeling(self.n, self.m, self.l)
        df = pd.concat([self.dmu.reset_index(drop=True), pd.DataFrame(result).reset_index(drop=True)], axis=1)
        df.columns = label_list
        df = self.BenchMarking(self.n, self.dmu, df)
        df.to_csv(self.output_file_name, index=False)


class SBM_CRS:
    def __init__(self, input_file_name, output_file_name, n_output):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.n_output = n_output
        self.raw_data = pd.read_csv(input_file_name).iloc[:, 1:]
        self.dmu = pd.read_csv(input_file_name).iloc[:, 0]
        self.l = n_output
        self.m = len(self.raw_data.columns) - self.l
        self.n = len(self.raw_data)
        self.front_Y = self.raw_data.transpose().iloc[:self.l, :]
        self.front_X = self.raw_data.transpose().iloc[self.l:, :]

    def GetObjCoeff(m, n, l, X0):
        obj_coeff = [1]
        zero_n = list(np.repeat(0, n, axis=0))
        zero_l = list(np.repeat(0, l, axis=0))
        input_weighted = (-1 / m) * (1 / X0)
        obj_coeff.extend(zero_n)
        obj_coeff.extend(list(input_weighted))
        obj_coeff.extend(zero_l)
        return obj_coeff

    def DefineVar(obj_coeff):
        n_var = len(obj_coeff)
        return LpVariable.dicts("V", list(range(n_var)), lowBound=0, cat="Continuous")

    def SetObjCoeff(obj_coeff, variables, problem):
        obj_func = sum(coeff * variables[idx] for idx, coeff in enumerate(obj_coeff))
        problem += obj_func
        return problem

    def LinearConstraint(m, n, l, Y0, variables):
        con_coeff = [1]
        zero_n = list(np.repeat(0, n, axis=0))
        zero_m = list(np.repeat(0, m, axis=0))
        output_weighted = [(1 / l) * (1 / Y0)]
        con_coeff.extend(zero_n)
        con_coeff.extend(zero_m)
        con_coeff.extend(output_weighted)
        constraint = sum(coeff * variables[idx] for idx, coeff in enumerate(con_coeff))
        return constraint

    def InputConstraint(X0, front_X, m, l, variables):
        neg_X0 = -X0
        input_const_list = []
        diag_mat = np.identity(m)
        zero_mat = np.zeros((m, l))
        const_mat = pd.concat([
            neg_X0.reset_index(drop=True),
            front_X.reset_index(drop=True),
            pd.DataFrame(diag_mat).reset_index(drop=True),
            pd.DataFrame(zero_mat).reset_index(drop=True)
        ], axis=1).to_numpy()

        for each_input in const_mat:
            constraint = sum(coeff * variables[idx] for idx, coeff in enumerate(each_input))
            input_const_list.append(constraint)

        return input_const_list

    def OutputConstraint(Y0, front_Y, m, l, variables):
        neg_Y0 = -Y0
        output_const_list = []
        diag_mat = np.identity(l)
        zero_mat = np.zeros((l, m))
        const_mat = pd.concat([
            neg_Y0.reset_index(drop=True),
            front_Y.reset_index(drop=True),
            pd.DataFrame(zero_mat).reset_index(drop=True),
            pd.DataFrame(diag_mat).reset_index(drop=True)
        ], axis=1).to_numpy()

        for each_output in const_mat:
            constraint = sum(coeff * variables[idx] for idx, coeff in enumerate(each_output))
            output_const_list.append(constraint)

        return output_const_list

    def AddConstraints(self, X0, Y0, m, n, l, front_X, front_Y, variables, problem):
        linear_coeff = self.LinearConstraint(m, n, l, Y0, variables)
        problem += linear_coeff == 1

        input_const = self.InputConstraint(X0, front_X, m, l, variables)
        for each_input_const in input_const:
            problem += each_input_const == 0

        output_const = self.OutputConstraint(Y0, front_Y, m, l, variables)
        for each_output_const in output_const:
            problem += each_output_const == 0

        return problem

    def Labeling(n, m, l):
        label_list = ["dmu", "sbm.vrs_eff"]
        label_list += [f"lambda_{idx + 1}" for idx in range(n)]
        label_list += [f"x_slack_{idx + 1}" for idx in range(m)]
        label_list += [f"y_slack_{idx + 1}" for idx in range(l)]
        return label_list

    def BenchMarking(n, dmu, df):
        benchmark_list = []
        for row_idx in range(n):
            benchmarks = [
                dmu[col_idx]
                for col_idx, each_lambda in enumerate(df.iloc[row_idx, 2:n + 2])
                if each_lambda > 0 and dmu[col_idx] != dmu[row_idx]
            ]
            benchmark_list.append(",".join(benchmarks))
        df["benchmark"] = benchmark_list
        return df

    def Solve(self):
        result = []
        for idx in range(len(self.raw_data)):
            X0 = self.front_X.iloc[:, idx]
            Y0 = self.front_Y.iloc[:, idx]
            problem = LpProblem("Slack-based Model", LpMinimize)
            obj_coeff = self.GetObjCoeff(self.m, self.n, self.l, X0)
            variables = self.DefineVar(obj_coeff)
            problem = self.SetObjCoeff(obj_coeff, variables, problem)
            problem = self.AddConstraints(X0, Y0, self.m, self.n, self.l, self.front_X, self.front_Y, variables, problem)

            problem.solve(GLPK(msg=0))
            eff = pulp.value(problem.objective)
            solution_list = [
                variable.varValue / variables[0].varValue
                for idx, variable in variables.items()
                if idx != 0
            ]
            result.append([eff] + solution_list)

        label_list = self.Labeling(self.n, self.m, self.l)
        df = pd.concat([self.dmu.reset_index(drop=True), pd.DataFrame(result).reset_index(drop=True)], axis=1)
        df.columns = label_list
        df = self.BenchMarking(self.n, self.dmu, df)
        df.to_csv(self.output_file_name, index=False)

if __name__ == "__main__":

    input_file_name = "input.csv"  # Replace with your actual input file
    output_file_name_vrs = "output_vrs.csv"  # Output for VRS
    output_file_name_crs = "output_crs.csv"  # Output for CRS
    n_output = 1  # Specify the number of outputs
    
    # Solve for VRS
    sbm_vrs = SBM_VRS(input_file_name=input_file_name, output_file_name=output_file_name_vrs, n_output=n_output)
    sbm_vrs.Solve()
    print(f"VRS Results saved to {output_file_name_vrs}")

    # Solve for CRS
    sbm_crs = SBM_CRS(input_file_name=input_file_name, output_file_name=output_file_name_crs, n_output=n_output)
    sbm_crs.Solve()
    print(f"CRS Results saved to {output_file_name_crs}")
