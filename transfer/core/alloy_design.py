"""
合金设计和优化模块
包含遗传算法优化、合金生成等功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
from typing import List, Tuple, Dict, Optional, Callable, Any
import os
from parameter_cal import parameter_calculation


class AlloyGenerator:
    """合金生成器"""
    
    def __init__(self, element_list: List[str], random_state: int = 42):
        self.element_list = element_list
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    def generate_random_alloy(self, num_samples: int, 
                            composition_constraints: Dict[str, Tuple[float, float]],
                            process_params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """生成随机合金成分"""
        
        data = []
        element_data = []
        
        for _ in range(num_samples * 10):  # 生成更多样本以确保满足约束
            # 生成随机成分
            composition = self._generate_composition(composition_constraints)
            
            if composition is not None:
                element_data.append(composition)
                
                # 添加工艺参数
                alloy_data = list(composition)
                for param_name, param_value in process_params.items():
                    if isinstance(param_value, list):
                        alloy_data.append(random.choice(param_value))
                    else:
                        alloy_data.append(param_value)
                
                data.append(alloy_data)
                
                if len(data) >= num_samples:
                    break
        
        return np.array(data), np.array(element_data)
    
    def _generate_composition(self, constraints: Dict[str, Tuple[float, float]]) -> Optional[np.ndarray]:
        """生成满足约束的成分"""
        
        composition = np.zeros(len(self.element_list))
        
        # 随机选择非零元素
        num_elements = random.randint(3, len(self.element_list))
        element_indices = random.sample(range(len(self.element_list)), num_elements)
        
        # 生成随机值
        for i, idx in enumerate(element_indices[:-1]):
            element_name = self.element_list[idx]
            min_val, max_val = constraints.get(element_name, (0, 50))
            composition[idx] = random.uniform(min_val, max_val)
        
        # 最后一个元素确保总和为100
        last_idx = element_indices[-1]
        remaining = 100 - np.sum(composition)
        
        if remaining < 0:
            return None
        
        element_name = self.element_list[last_idx]
        min_val, max_val = constraints.get(element_name, (0, 50))
        
        if min_val <= remaining <= max_val:
            composition[last_idx] = remaining
            return composition
        else:
            return None


class GeneticOptimizer:
    """遗传算法优化器"""
    
    def __init__(self, element_list: List[str], 
                 composition_constraints: Dict[str, Tuple[float, float]],
                 objective_function: Callable,
                 random_state: int = 42):
        
        self.element_list = element_list
        self.composition_constraints = composition_constraints
        self.objective_function = objective_function
        self.random_state = random_state
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 创建DEAP框架
        self._setup_deap()
    
    def _setup_deap(self):
        """设置DEAP框架"""
        
        # 创建适应度类
        creator.create("FitnessMulti", base.Fitness, weights=(1, -1))  # 最大化强度，最小化密度
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        # 初始化工具箱
        self.toolbox = base.Toolbox()
        
        # 注册函数
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)
    
    def _create_individual(self):
        """创建个体"""
        
        while True:
            individual = []
            
            for element in self.element_list:
                min_val, max_val = self.composition_constraints.get(element, (0, 50))
                individual.append(random.uniform(min_val, max_val))
            
            # 归一化
            individual = self._normalize_individual(individual)
            
            if individual is not None:
                return creator.Individual(individual)
    
    def _normalize_individual(self, individual: List[float]) -> Optional[List[float]]:
        """归一化个体"""
        
        # 确保在约束范围内
        normalized = []
        for i, element in enumerate(self.element_list):
            min_val, max_val = self.composition_constraints.get(element, (0, 50))
            normalized.append(max(min_val, min(max_val, individual[i])))
        
        # 归一化到总和为100
        total = sum(normalized)
        if total > 0:
            normalized = [x / total * 100 for x in normalized]
            return normalized
        else:
            return None
    
    def _evaluate(self, individual):
        """评估个体"""
        
        # 确保归一化
        individual = self._normalize_individual(individual)
        
        if individual is None:
            return (0, 1000)  # 惩罚无效个体
        
        return self.objective_function(individual)
    
    def optimize(self, population_size: int = 100, generations: int = 50,
                crossover_prob: float = 0.8, mutation_prob: float = 0.1,
                verbose: bool = True) -> Dict[str, Any]:
        """执行优化"""
        
        # 创建初始种群
        population = self.toolbox.population(n=population_size)
        
        # 记录进化历史
        evolution_history = {
            'generations': [],
            'best_fitness': [],
            'pareto_front': []
        }
        
        # 进化过程
        for gen in range(generations):
            if verbose:
                print(f"第 {gen + 1}/{generations} 代")
            
            # 选择
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # 交叉
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # 变异
            for mutant in offspring:
                if random.random() < mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # 归一化
            for ind in offspring:
                ind[:] = self._normalize_individual(ind)
            
            # 评估
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # 替换种群
            population[:] = offspring
            
            # 记录历史
            best_individual = tools.selBest(population, 1)[0]
            evolution_history['generations'].append(gen + 1)
            evolution_history['best_fitness'].append(best_individual.fitness.values)
            
            # Pareto前沿
            pareto_front = tools.sortNondominated(population, len(population))[0]
            evolution_history['pareto_front'].append(pareto_front)
            
            if verbose:
                print(f"最佳适应度: {best_individual.fitness.values}")
        
        return {
            'final_population': population,
            'best_individual': best_individual,
            'evolution_history': evolution_history
        }
    
    def plot_evolution(self, evolution_history: Dict[str, Any], 
                     save_path: Optional[str] = None):
        """绘制进化过程"""
        
        generations = evolution_history['generations']
        best_fitness = evolution_history['best_fitness']
        
        plt.figure(figsize=(12, 5))
        
        # 绘制强度进化
        plt.subplot(1, 2, 1)
        strength_values = [fit[0] for fit in best_fitness]
        plt.plot(generations, strength_values, 'b-', linewidth=2)
        plt.xlabel('代数')
        plt.ylabel('强度')
        plt.title('强度进化过程')
        plt.grid(True, alpha=0.3)
        
        # 绘制密度进化
        plt.subplot(1, 2, 2)
        density_values = [fit[1] for fit in best_fitness]
        plt.plot(generations, density_values, 'r-', linewidth=2)
        plt.xlabel('代数')
        plt.ylabel('密度')
        plt.title('密度进化过程')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pareto_front(self, pareto_front: List, 
                         save_path: Optional[str] = None):
        """绘制Pareto前沿"""
        
        strength_values = [ind.fitness.values[0] for ind in pareto_front]
        density_values = [ind.fitness.values[1] for ind in pareto_front]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(density_values, strength_values, c='blue', alpha=0.7, s=50)
        plt.xlabel('密度', fontsize=12)
        plt.ylabel('强度', fontsize=12)
        plt.title('Pareto前沿', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, results: Dict[str, Any], save_dir: str):
        """保存优化结果"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存最佳个体
        best_individual = results['best_individual']
        best_composition = [x * 100 for x in best_individual]  # 转换为百分比
        
        best_df = pd.DataFrame([best_composition], columns=self.element_list)
        best_df['strength'] = [best_individual.fitness.values[0]]
        best_df['density'] = [best_individual.fitness.values[1]]
        best_df.to_csv(os.path.join(save_dir, 'best_individual.csv'), index=False)
        
        # 保存Pareto前沿
        pareto_front = results['evolution_history']['pareto_front'][-1]
        pareto_data = []
        
        for ind in pareto_front:
            composition = [x * 100 for x in ind]
            pareto_data.append(composition + list(ind.fitness.values))
        
        pareto_df = pd.DataFrame(pareto_data, 
                               columns=self.element_list + ['strength', 'density'])
        pareto_df.to_csv(os.path.join(save_dir, 'pareto_front.csv'), index=False)
        
        print(f"优化结果已保存到 {save_dir}")


class AlloyDesigner:
    """合金设计器"""
    
    def __init__(self, element_list: List[str], 
                 composition_constraints: Dict[str, Tuple[float, float]],
                 model_predictor: Callable):
        
        self.element_list = element_list
        self.composition_constraints = composition_constraints
        self.model_predictor = model_predictor
        self.generator = AlloyGenerator(element_list)
    
    def design_alloy(self, num_samples: int, 
                    process_params: Dict[str, Any],
                    optimization_target: str = 'strength') -> pd.DataFrame:
        """设计合金"""
        
        # 生成随机合金
        data, element_data = self.generator.generate_random_alloy(
            num_samples, self.composition_constraints, process_params
        )
        
        # 预测性能
        predictions = self.model_predictor(data)
        
        # 创建结果DataFrame
        columns = self.element_list + list(process_params.keys()) + ['predicted_property']
        result_df = pd.DataFrame(data, columns=columns)
        result_df['predicted_property'] = predictions.flatten()
        
        return result_df
    
    def optimize_alloy(self, population_size: int = 100, generations: int = 50,
                      process_params: Dict[str, Any]) -> Dict[str, Any]:
        """优化合金设计"""
        
        def objective_function(composition):
            """目标函数"""
            # 添加工艺参数
            alloy_data = list(composition)
            for param_name, param_value in process_params.items():
                if isinstance(param_value, list):
                    alloy_data.append(random.choice(param_value))
                else:
                    alloy_data.append(param_value)
            
            # 预测性能
            prediction = self.model_predictor(np.array([alloy_data]))
            
            # 计算密度
            df_parameter = parameter_calculation(np.array([composition]), len(composition))
            density = df_parameter['density'].iloc[0]
            
            return prediction[0][0], density
        
        # 创建优化器
        optimizer = GeneticOptimizer(
            self.element_list, 
            self.composition_constraints, 
            objective_function
        )
        
        # 执行优化
        results = optimizer.optimize(population_size, generations)
        
        return results
