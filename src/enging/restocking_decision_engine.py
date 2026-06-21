import numpy as np

class RestockingDecisionEngine:
    def __init__(self, lead_time=3, safety_stock=50):
        self.lead_time = lead_time
        self.safety_stock = safety_stock

    def generate_recommendation(self, current_inventory, predicted_demand):
        predicted_demand = np.array(predicted_demand)
        
        demand_during_lead_time = np.sum(predicted_demand[:self.lead_time])
        reorder_point = demand_during_lead_time + self.safety_stock
        
        restock_recommended = current_inventory <= reorder_point
        
        target_stock_level = np.sum(predicted_demand) + self.safety_stock
        recommended_quantity = int(max(0, target_stock_level - current_inventory)) if restock_recommended else 0
        
        return {
            "demand_during_lead_time": float(demand_during_lead_time),
            "reorder_point": float(reorder_point),
            "restock_recommended": bool(restock_recommended),
            "recommended_quantity": recommended_quantity
        }
