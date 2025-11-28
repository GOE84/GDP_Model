"""
GDP Model and Visualization
โมเดลสำหรับคำนวณและแสดงผล GDP (Gross Domestic Product)

GDP = C + I + G + (X - M)
โดยที่:
C = การบริโภคของครัวเรือน (Consumption)
I = การลงทุน (Investment)
G = การใช้จ่ายของรัฐบาล (Government Spending)
X = การส่งออก (Exports)
M = การนำเข้า (Imports)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# ตั้งค่าให้ matplotlib รองรับภาษาไทย
plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class GDPModel:
    """โมเดลสำหรับคำนวณ GDP"""
    
    def __init__(self):
        self.data = None
        self.gdp_values = None
    
    def calculate_gdp(self, consumption, investment, government_spending, exports, imports):
        """
        คำนวณ GDP จากสูตร GDP = C + I + G + (X - M)
        
        Parameters:
        -----------
        consumption : float or array
            การบริโภคของครัวเรือน
        investment : float or array
            การลงทุน
        government_spending : float or array
            การใช้จ่ายของรัฐบาล
        exports : float or array
            การส่งออก
        imports : float or array
            การนำเข้า
        
        Returns:
        --------
        gdp : float or array
            ค่า GDP ที่คำนวณได้
        """
        net_exports = exports - imports
        gdp = consumption + investment + government_spending + net_exports
        return gdp
    
    def generate_sample_data(self, years=10, start_year=2015):
        """
        สร้างข้อมูลตัวอย่างสำหรับการคำนวณ GDP
        
        Parameters:
        -----------
        years : int
            จำนวนปีที่ต้องการสร้างข้อมูล
        start_year : int
            ปีเริ่มต้น
        """
        np.random.seed(42)
        
        years_list = list(range(start_year, start_year + years))
        
        # สร้างข้อมูลที่มีแนวโน้มเพิ่มขึ้นพร้อมความผันแปร
        base_consumption = 5000
        base_investment = 1500
        base_gov_spending = 2000
        base_exports = 1800
        base_imports = 1600
        
        data = {
            'Year': years_list,
            'Consumption': [base_consumption + i*200 + np.random.normal(0, 100) for i in range(years)],
            'Investment': [base_investment + i*80 + np.random.normal(0, 50) for i in range(years)],
            'Government_Spending': [base_gov_spending + i*100 + np.random.normal(0, 60) for i in range(years)],
            'Exports': [base_exports + i*90 + np.random.normal(0, 70) for i in range(years)],
            'Imports': [base_imports + i*85 + np.random.normal(0, 65) for i in range(years)]
        }
        
        self.data = pd.DataFrame(data)
        
        # คำนวณ GDP
        self.data['GDP'] = self.calculate_gdp(
            self.data['Consumption'],
            self.data['Investment'],
            self.data['Government_Spending'],
            self.data['Exports'],
            self.data['Imports']
        )
        
        self.data['Net_Exports'] = self.data['Exports'] - self.data['Imports']
        
        return self.data
    
    def plot_gdp_trend(self, save_path='gdp_trend.png'):
        """
        แสดงกราฟแนวโน้ม GDP
        """
        if self.data is None:
            raise ValueError("ไม่มีข้อมูล กรุณาสร้างข้อมูลก่อนด้วย generate_sample_data()")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Year'], self.data['GDP'], marker='o', linewidth=2, 
                markersize=8, color='#2E86AB', label='GDP')
        
        plt.title('GDP Trend Over Time / แนวโน้ม GDP ตามช่วงเวลา', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year / ปี', fontsize=12, fontweight='bold')
        plt.ylabel('GDP (Billion) / GDP (พันล้าน)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11)
        
        # เพิ่มค่าบนจุดข้อมูล
        for i, row in self.data.iterrows():
            plt.annotate(f'{row["GDP"]:.0f}', 
                        xy=(row['Year'], row['GDP']),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ บันทึกกราฟที่: {save_path}")
        plt.show()
    
    def plot_gdp_components(self, save_path='gdp_components.png'):
        """
        แสดงกราฟองค์ประกอบของ GDP
        """
        if self.data is None:
            raise ValueError("ไม่มีข้อมูล กรุณาสร้างข้อมูลก่อนด้วย generate_sample_data()")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        years = self.data['Year']
        width = 0.6
        
        # สร้าง stacked bar chart
        p1 = ax.bar(years, self.data['Consumption'], width, label='Consumption / การบริโภค', color='#A23B72')
        p2 = ax.bar(years, self.data['Investment'], width, bottom=self.data['Consumption'],
                   label='Investment / การลงทุน', color='#F18F01')
        p3 = ax.bar(years, self.data['Government_Spending'], width,
                   bottom=self.data['Consumption'] + self.data['Investment'],
                   label='Government Spending / การใช้จ่ายรัฐ', color='#C73E1D')
        p4 = ax.bar(years, self.data['Net_Exports'], width,
                   bottom=self.data['Consumption'] + self.data['Investment'] + self.data['Government_Spending'],
                   label='Net Exports / การส่งออกสุทธิ', color='#6A994E')
        
        ax.set_title('GDP Components / องค์ประกอบของ GDP', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Year / ปี', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value (Billion) / มูลค่า (พันล้าน)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ บันทึกกราฟที่: {save_path}")
        plt.show()
    
    def plot_growth_rate(self, save_path='gdp_growth_rate.png'):
        """
        แสดงกราฟอัตราการเติบโตของ GDP
        """
        if self.data is None:
            raise ValueError("ไม่มีข้อมูล กรุณาสร้างข้อมูลก่อนด้วย generate_sample_data()")
        
        # คำนวณอัตราการเติบโต
        growth_rate = self.data['GDP'].pct_change() * 100
        
        plt.figure(figsize=(12, 6))
        colors = ['#06A77D' if x >= 0 else '#D62828' for x in growth_rate[1:]]
        plt.bar(self.data['Year'][1:], growth_rate[1:], color=colors, alpha=0.7, edgecolor='black')
        
        plt.title('GDP Growth Rate / อัตราการเติบโตของ GDP', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Year / ปี', fontsize=12, fontweight='bold')
        plt.ylabel('Growth Rate (%) / อัตราการเติบโต (%)', fontsize=12, fontweight='bold')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # เพิ่มค่าบนแท่ง
        for i, (year, rate) in enumerate(zip(self.data['Year'][1:], growth_rate[1:])):
            plt.annotate(f'{rate:.1f}%', 
                        xy=(year, rate),
                        xytext=(0, 5 if rate >= 0 else -15), 
                        textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ บันทึกกราฟที่: {save_path}")
        plt.show()
    
    def plot_all_components_trends(self, save_path='all_components_trends.png'):
        """
        แสดงกราฟแนวโน้มของทุกองค์ประกอบ
        """
        if self.data is None:
            raise ValueError("ไม่มีข้อมูล กรุณาสร้างข้อมูลก่อนด้วย generate_sample_data()")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('GDP Components Trends / แนวโน้มองค์ประกอบของ GDP', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        components = [
            ('Consumption', 'การบริโภค', '#A23B72'),
            ('Investment', 'การลงทุน', '#F18F01'),
            ('Government_Spending', 'การใช้จ่ายรัฐ', '#C73E1D'),
            ('Exports', 'การส่งออก', '#2E86AB'),
            ('Imports', 'การนำเข้า', '#E63946'),
            ('GDP', 'GDP', '#06A77D')
        ]
        
        for idx, (component, thai_name, color) in enumerate(components):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            ax.plot(self.data['Year'], self.data[component], 
                   marker='o', linewidth=2, markersize=6, color=color)
            ax.set_title(f'{component} / {thai_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year / ปี', fontsize=10)
            ax.set_ylabel('Value (Billion) / มูลค่า', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ บันทึกกราฟที่: {save_path}")
        plt.show()
    
    def print_summary(self):
        """
        แสดงสรุปข้อมูล GDP
        """
        if self.data is None:
            raise ValueError("ไม่มีข้อมูล กรุณาสร้างข้อมูลก่อนด้วย generate_sample_data()")
        
        print("\n" + "="*70)
        print("GDP SUMMARY / สรุปข้อมูล GDP".center(70))
        print("="*70)
        print(f"\nช่วงเวลา: {self.data['Year'].min()} - {self.data['Year'].max()}")
        print(f"\nGDP เฉลี่ย: {self.data['GDP'].mean():.2f} พันล้าน")
        print(f"GDP สูงสุด: {self.data['GDP'].max():.2f} พันล้าน (ปี {self.data.loc[self.data['GDP'].idxmax(), 'Year']})")
        print(f"GDP ต่ำสุด: {self.data['GDP'].min():.2f} พันล้าน (ปี {self.data.loc[self.data['GDP'].idxmin(), 'Year']})")
        
        growth_rate = self.data['GDP'].pct_change() * 100
        print(f"\nอัตราการเติบโตเฉลี่ย: {growth_rate[1:].mean():.2f}%")
        print(f"อัตราการเติบโตสูงสุด: {growth_rate[1:].max():.2f}% (ปี {self.data.loc[growth_rate[1:].idxmax(), 'Year']})")
        
        print("\n" + "-"*70)
        print("ข้อมูลรายปี:".center(70))
        print("-"*70)
        print(self.data.to_string(index=False))
        print("="*70 + "\n")


def main():
    """
    ฟังก์ชันหลักสำหรับรันโมเดล GDP
    """
    print("\n🚀 เริ่มต้นโมเดล GDP Model")
    print("="*70)
    
    # สร้างโมเดล
    model = GDPModel()
    
    # สร้างข้อมูลตัวอย่าง (10 ปี เริ่มจากปี 2015)
    print("\n📊 กำลังสร้างข้อมูลตัวอย่าง...")
    data = model.generate_sample_data(years=10, start_year=2015)
    print("✓ สร้างข้อมูลเสร็จสิ้น")
    
    # แสดงสรุปข้อมูล
    model.print_summary()
    
    # สร้างกราฟต่างๆ
    print("\n📈 กำลังสร้างกราฟ...")
    print("-"*70)
    
    model.plot_gdp_trend('gdp_trend.png')
    model.plot_gdp_components('gdp_components.png')
    model.plot_growth_rate('gdp_growth_rate.png')
    model.plot_all_components_trends('all_components_trends.png')
    
    print("\n" + "="*70)
    print("✅ เสร็จสิ้นการสร้างกราฟทั้งหมด!")
    print("="*70)
    print("\nไฟล์ที่สร้าง:")
    print("  1. gdp_trend.png - กราฟแนวโน้ม GDP")
    print("  2. gdp_components.png - กราฟองค์ประกอบของ GDP")
    print("  3. gdp_growth_rate.png - กราฟอัตราการเติบโต")
    print("  4. all_components_trends.png - กราฟแนวโน้มทุกองค์ประกอบ")
    print("\n")


if __name__ == "__main__":
    main()
