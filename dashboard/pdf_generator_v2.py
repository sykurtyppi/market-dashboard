"""
PDF Report Generator v2 - With Charts Support
"""

import os
import logging
from datetime import datetime
from pathlib import Path
import plotly.io as pio
import pandas as pd
from fpdf import FPDF

logger = logging.getLogger(__name__)


class MarketReportPDF(FPDF):
    """Enhanced PDF with better styling"""
    
    def __init__(self):
        super().__init__()
        self.report_date = datetime.now().strftime("%B %d, %Y")
    
    def header(self):
        """Professional header"""
        # Blue header bar
        self.set_fill_color(31, 119, 180)
        self.rect(0, 0, 210, 25, 'F')
        
        # White text
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 20)
        self.set_y(8)
        self.cell(0, 10, 'Market Risk Dashboard', 0, 1, 'C')
        self.set_font('Arial', '', 11)
        self.cell(0, 5, f'Weekly Report - {self.report_date}', 0, 1, 'C')
        
        # Reset
        self.set_text_color(0, 0, 0)
        self.ln(10)
    
    def footer(self):
        """Enhanced footer"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        
        # Left: Page number
        self.cell(60, 10, f'Page {self.page_no()}', 0, 0, 'L')
        
        # Center: Disclaimer
        self.cell(90, 10, 'Not Financial Advice', 0, 0, 'C')
        
        # Right: Timestamp
        self.cell(40, 10, datetime.now().strftime("%Y-%m-%d"), 0, 0, 'R')
    
    def add_section_header(self, title):
        """Add section header with line"""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(31, 119, 180)
        self.cell(0, 10, title, 0, 1, 'L')
        
        # Underline
        self.set_draw_color(31, 119, 180)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        
        self.set_text_color(0, 0, 0)
        self.ln(5)
    
    def add_metric_box(self, title, value, status='neutral', x=None):
        """Enhanced metric box with border"""
        colors = {
            'good': (76, 175, 80),
            'warning': (255, 152, 0),
            'bad': (244, 67, 54),
            'neutral': (158, 158, 158)
        }
        
        color = colors.get(status, colors['neutral'])
        
        if x:
            self.set_x(x)
        
        start_x = self.get_x()
        start_y = self.get_y()
        
        # Background
        self.set_fill_color(250, 250, 250)
        self.rect(start_x, start_y, 45, 20, 'F')
        
        # Colored left border
        self.set_fill_color(*color)
        self.rect(start_x, start_y, 3, 20, 'F')
        
        # Title
        self.set_xy(start_x + 5, start_y + 3)
        self.set_font('Arial', '', 8)
        self.set_text_color(100, 100, 100)
        self.cell(40, 4, title, 0, 0, 'L')
        
        # Value
        self.set_xy(start_x + 5, start_y + 10)
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 0, 0)
        self.cell(40, 6, str(value), 0, 0, 'L')
        
        return start_x + 48  # Return next x position
    
    def add_insight_box(self, text, box_type='info'):
        """Add colored insight box"""
        colors = {
            'info': (33, 150, 243),
            'success': (76, 175, 80),
            'warning': (255, 152, 0),
            'danger': (244, 67, 54)
        }
        
        color = colors.get(box_type, colors['info'])
        
        # Draw box
        start_y = self.get_y()
        self.set_fill_color(color[0], color[1], color[2])
        self.rect(10, start_y, 4, 15, 'F')  # Left colored bar
        
        self.set_fill_color(245, 245, 245)
        self.rect(14, start_y, 186, 15, 'F')  # Gray background
        
        # Add text
        self.set_xy(18, start_y + 3)
        self.set_font('Arial', '', 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(176, 4.5, text)
        
        self.ln(3)


class PDFReportGenerator:
    """Enhanced PDF generator with chart support"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("/tmp/market_dashboard_charts")
        self.temp_dir.mkdir(exist_ok=True)
    
    def save_chart(self, fig, name: str) -> str:
        """Save Plotly chart as high-res PNG"""
        filepath = self.temp_dir / f"{name}.png"
        try:
            pio.write_image(
                fig, 
                str(filepath), 
                width=1200, 
                height=500, 
                scale=2
            )
            logger.info(f"Saved chart: {name}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Chart save failed {name}: {e}")
            return None
    
    def generate_report(
        self,
        snapshot: dict,
        vrp_data: dict,
        charts: dict = None,
        insights: list = None,
        filename: str = None
    ) -> str:
        """
        Generate enhanced PDF report
        
        Args:
            snapshot: Market snapshot data
            vrp_data: VRP analysis
            charts: Dict of Plotly figures {name: fig}
            insights: List of insight strings
            filename: Custom filename
        """
        if filename is None:
            filename = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        
        output_path = self.output_dir / filename
        
        pdf = MarketReportPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        
        # ============ PAGE 1: DASHBOARD ============
        pdf.add_page()
        
        # Key metrics in grid
        pdf.add_section_header("MARKET DASHBOARD")
        
        y_pos = pdf.get_y()
        
        # Row 1
        x = 10
        if snapshot and snapshot.get('credit_spread_hy'):
            hy = snapshot['credit_spread_hy'] * 100
            status = 'good' if hy < 300 else 'warning' if hy < 450 else 'bad'
            x = pdf.add_metric_box("HYG Spread", f"{hy:.0f} bps", status, x)
        
        if snapshot and snapshot.get('vix_spot'):
            vix = snapshot['vix_spot']
            status = 'good' if vix < 20 else 'warning' if vix < 30 else 'bad'
            x = pdf.add_metric_box("VIX", f"{vix:.1f}", status, x)
        
        if vrp_data and vrp_data.get('vrp'):
            vrp = vrp_data['vrp']
            status = 'good' if vrp > 4 else 'warning' if vrp > 0 else 'bad'
            x = pdf.add_metric_box("VRP", f"{vrp:+.2f}", status, x)
        
        if snapshot and snapshot.get('fear_greed_score'):
            fg = snapshot['fear_greed_score']
            status = 'bad' if fg < 25 else 'warning' if fg < 45 else 'good'
            pdf.add_metric_box("Fear/Greed", f"{fg:.0f}", status, x)
        
        pdf.ln(25)
        
        # Insights section
        if insights:
            pdf.add_section_header("MARKET INSIGHTS")
            for insight in insights:
                pdf.add_insight_box(insight, 'info')
        
        # ============ PAGE 2+: CHARTS ============
        if charts:
            for i, (chart_name, fig) in enumerate(charts.items()):
                if i % 2 == 0:  # 2 charts per page
                    pdf.add_page()
                
                img_path = self.save_chart(fig, chart_name)
                if img_path and os.path.exists(img_path):
                    pdf.add_section_header(chart_name.replace('_', ' ').title())
                    pdf.image(img_path, x=10, w=190)
                    pdf.ln(5)
        
        # ============ LAST PAGE: DATA TABLE ============
        pdf.add_page()
        pdf.add_section_header("DETAILED METRICS")
        
        if snapshot:
            data = [
                ("Report Date", snapshot.get('date', 'N/A')),
                ("HYG Spread", f"{snapshot.get('credit_spread_hy', 0)*100:.0f} bps"),
                ("LQD Spread", f"{snapshot.get('credit_spread_ig', 0)*100:.0f} bps"),
                ("10Y Treasury", f"{snapshot.get('treasury_10y', 0):.2f}%"),
                ("VIX", f"{snapshot.get('vix_spot', 0):.2f}"),
                ("Fear & Greed", f"{snapshot.get('fear_greed_score', 0):.0f}"),
                ("Market Breadth", f"{snapshot.get('market_breadth', 0)*100:.1f}%"),
            ]
            
            if vrp_data:
                data.extend([
                    ("VRP", f"{vrp_data.get('vrp', 0):+.2f}"),
                    ("Realized Vol", f"{vrp_data.get('realized_vol', 0):.2f}"),
                    ("VIX Regime", vrp_data.get('regime', 'N/A')),
                ])
            
            pdf.set_font('Arial', '', 9)
            pdf.set_fill_color(245, 245, 245)
            
            for i, (label, value) in enumerate(data):
                fill = i % 2 == 0
                pdf.cell(90, 7, label, 1, 0, 'L', fill)
                pdf.cell(100, 7, str(value), 1, 1, 'R', fill)
        
        # Save
        pdf.output(str(output_path))
        logger.info(f"PDF generated: {output_path}")
        
        return str(output_path)


def create_enhanced_sample():
    """Test enhanced version"""
    generator = PDFReportGenerator()
    
    snapshot = {
        'date': '2025-11-27',
        'credit_spread_hy': 3.10,
        'credit_spread_ig': 4.75,
        'treasury_10y': 4.25,
        'vix_spot': 17.1,
        'fear_greed_score': 18,
        'market_breadth': 0.909,
    }
    
    vrp = {
        'vrp': 1.73,
        'realized_vol': 15.37,
        'regime': 'Elevated',
    }
    
    insights = [
        "Extreme Fear (18) combined with positive VRP (+1.73) suggests a contrarian buying opportunity. Historically, readings below 25 have preceded strong rallies.",
        "Credit spreads remain neutral (HYG 310 bps), indicating no systemic stress despite equity market fear.",
        "Market breadth at 90.9% shows strong participation - this is a healthy sign for continuation.",
    ]
    
    output = generator.generate_report(
        snapshot=snapshot,
        vrp_data=vrp,
        charts={},  # Will add charts from dashboard later
        insights=insights,
        filename="enhanced_sample.pdf"
    )
    
    print(f"Enhanced report: {output}")
    return output


if __name__ == "__main__":
    create_enhanced_sample()
