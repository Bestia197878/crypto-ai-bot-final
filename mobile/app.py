"""
Kivy Mobile App for Crypto Trading AI
"""
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.properties import StringProperty, ObjectProperty, NumericProperty
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton, MDIconButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.list import MDList, OneLineListItem, TwoLineListItem
from kivymd.uix.toolbar import MDToolbar
from kivymd.uix.bottomnavigation import MDBottomNavigation, MDBottomNavigationItem
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog
from kivymd.uix.snackbar import Snackbar

import json
from datetime import datetime
from typing import Dict, List


class PortfolioScreen(MDScreen):
    """Portfolio overview screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = MDBoxLayout(orientation='vertical')
        
        # Header
        header = MDBoxLayout(size_hint_y=None, height=dp(150), padding=dp(20))
        header.md_bg_color = [0.1, 0.1, 0.2, 1]
        
        with header.canvas.before:
            Color(0.1, 0.1, 0.2, 1)
            Rectangle(pos=header.pos, size=header.size)
        
        header.bind(pos=self._update_rect, size=self._update_rect)
        
        # Portfolio value
        self.value_label = MDLabel(
            text="$10,000.00",
            font_style="H3",
            theme_text_color="Custom",
            text_color=[1, 1, 1, 1],
            halign="center"
        )
        
        self.return_label = MDLabel(
            text="+5.23% ($523.00)",
            font_style="Subtitle1",
            theme_text_color="Custom",
            text_color=[0, 1, 0, 1],
            halign="center"
        )
        
        header.add_widget(self.value_label)
        header.add_widget(self.return_label)
        
        layout.add_widget(header)
        
        # Stats grid
        stats_grid = MDGridLayout(cols=2, spacing=dp(10), padding=dp(20), size_hint_y=None, height=dp(200))
        
        self.stats_cards = []
        stats = [
            ("Cash", "$10,000.00"),
            ("Invested", "$0.00"),
            ("Open Positions", "0"),
            ("Win Rate", "0%")
        ]
        
        for title, value in stats:
            card = MDCard(orientation='vertical', padding=dp(15), size_hint_y=None, height=dp(80))
            card.add_widget(MDLabel(text=title, font_style="Caption", theme_text_color="Secondary"))
            card.add_widget(MDLabel(text=value, font_style="H6"))
            stats_grid.add_widget(card)
            self.stats_cards.append((title, card))
        
        layout.add_widget(stats_grid)
        
        # Positions list
        layout.add_widget(MDLabel(text="Open Positions", font_style="H6", padding=dp(20)))
        
        self.positions_list = MDList()
        scroll = ScrollView()
        scroll.add_widget(self.positions_list)
        layout.add_widget(scroll)
        
        self.add_widget(layout)
        
        # Update timer
        Clock.schedule_interval(self.update_data, 5)
    
    def _update_rect(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0.1, 0.1, 0.2, 1)
            Rectangle(pos=instance.pos, size=instance.size)
    
    def update_data(self, dt):
        """Update portfolio data"""
        # In real app, fetch from API
        pass


class TradingScreen(MDScreen):
    """Trading screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
        
        # Symbol selector
        layout.add_widget(MDLabel(text="Trading Pair", font_style="H6"))
        
        self.symbol_input = MDTextField(
            text="BTCUSDT",
            hint_text="Symbol"
        )
        layout.add_widget(self.symbol_input)
        
        # Current price
        self.price_label = MDLabel(
            text="Current Price: $45,234.50",
            font_style="H5",
            halign="center"
        )
        layout.add_widget(self.price_label)
        
        # Price change
        self.change_label = MDLabel(
            text="+2.34%",
            theme_text_color="Custom",
            text_color=[0, 1, 0, 1],
            halign="center"
        )
        layout.add_widget(self.change_label)
        
        # Quantity input
        layout.add_widget(MDLabel(text="Quantity", font_style="H6"))
        self.quantity_input = MDTextField(
            text="0.1",
            hint_text="Quantity"
        )
        layout.add_widget(self.quantity_input)
        
        # Order value
        self.order_value_label = MDLabel(
            text="Order Value: $4,523.45",
            font_style="Subtitle1"
        )
        layout.add_widget(self.order_value_label)
        
        # Buy/Sell buttons
        button_layout = MDBoxLayout(size_hint_y=None, height=dp(60), spacing=dp(20))
        
        self.buy_button = MDRaisedButton(
            text="BUY",
            md_bg_color=[0, 0.8, 0, 1],
            size_hint=(0.5, 1)
        )
        self.buy_button.bind(on_press=self.on_buy)
        
        self.sell_button = MDRaisedButton(
            text="SELL",
            md_bg_color=[0.8, 0, 0, 1],
            size_hint=(0.5, 1)
        )
        self.sell_button.bind(on_press=self.on_sell)
        
        button_layout.add_widget(self.buy_button)
        button_layout.add_widget(self.sell_button)
        
        layout.add_widget(button_layout)
        
        # AI Prediction
        layout.add_widget(MDLabel(text="AI Prediction", font_style="H6"))
        
        prediction_card = MDCard(orientation='vertical', padding=dp(15))
        self.prediction_label = MDLabel(
            text="BUY - 85% Confidence",
            font_style="H5",
            theme_text_color="Custom",
            text_color=[0, 1, 0, 1]
        )
        prediction_card.add_widget(self.prediction_label)
        layout.add_widget(prediction_card)
        
        self.add_widget(layout)
        
        Clock.schedule_interval(self.update_price, 3)
    
    def update_price(self, dt):
        """Update price display"""
        pass
    
    def on_buy(self, instance):
        """Handle buy button"""
        Snackbar(text="Buy order placed!").open()
    
    def on_sell(self, instance):
        """Handle sell button"""
        Snackbar(text="Sell order placed!").open()


class AgentsScreen(MDScreen):
    """AI Agents screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = MDBoxLayout(orientation='vertical')
        
        layout.add_widget(MDLabel(text="AI Agents", font_style="H5", padding=dp(20)))
        
        self.agents_list = MDList()
        
        agents = [
            ("Super DQN", "Active", "85% accuracy"),
            ("Super Transformer", "Active", "82% accuracy"),
            ("LSTM Agent", "Training", "78% accuracy"),
            ("Super Ensemble", "Active", "89% accuracy"),
            ("Self-Learning", "Active", "87% accuracy")
        ]
        
        for name, status, accuracy in agents:
            item = TwoLineListItem(
                text=name,
                secondary_text=f"{status} - {accuracy}"
            )
            self.agents_list.add_widget(item)
        
        scroll = ScrollView()
        scroll.add_widget(self.agents_list)
        layout.add_widget(scroll)
        
        self.add_widget(layout)


class SettingsScreen(MDScreen):
    """Settings screen"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()
    
    def build_ui(self):
        layout = MDBoxLayout(orientation='vertical', padding=dp(20), spacing=dp(10))
        
        layout.add_widget(MDLabel(text="Settings", font_style="H5"))
        
        # API Keys
        layout.add_widget(MDLabel(text="API Configuration", font_style="H6"))
        
        self.api_key_input = MDTextField(
            hint_text="API Key",
            password=True
        )
        layout.add_widget(self.api_key_input)
        
        self.secret_input = MDTextField(
            hint_text="Secret Key",
            password=True
        )
        layout.add_widget(self.secret_input)
        
        # Trading settings
        layout.add_widget(MDLabel(text="Trading Settings", font_style="H6"))
        
        self.position_size_input = MDTextField(
            text="10",
            hint_text="Position Size (%)"
        )
        layout.add_widget(self.position_size_input)
        
        self.stop_loss_input = MDTextField(
            text="2",
            hint_text="Stop Loss (%)"
        )
        layout.add_widget(self.stop_loss_input)
        
        self.take_profit_input = MDTextField(
            text="4",
            hint_text="Take Profit (%)"
        )
        layout.add_widget(self.take_profit_input)
        
        # Save button
        save_button = MDRaisedButton(
            text="Save Settings",
            size_hint=(1, None),
            height=dp(50)
        )
        save_button.bind(on_press=self.save_settings)
        layout.add_widget(save_button)
        
        self.add_widget(layout)
    
    def save_settings(self, instance):
        """Save settings"""
        Snackbar(text="Settings saved!").open()


class CryptoTradingMobileApp(MDApp):
    """Main mobile application"""
    
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Light"
        
        # Create screen manager
        sm = ScreenManager()
        
        # Add screens
        sm.add_widget(PortfolioScreen(name='portfolio'))
        sm.add_widget(TradingScreen(name='trading'))
        sm.add_widget(AgentsScreen(name='agents'))
        sm.add_widget(SettingsScreen(name='settings'))
        
        # Main layout with bottom navigation
        layout = MDBoxLayout(orientation='vertical')
        
        # Screen manager
        layout.add_widget(sm)
        
        # Bottom navigation
        bottom_nav = MDBottomNavigation()
        
        bottom_nav.add_widget(
            MDBottomNavigationItem(
                name='portfolio',
                text='Portfolio',
                icon='chart-pie',
                on_tab_press=lambda x: setattr(sm, 'current', 'portfolio')
            )
        )
        
        bottom_nav.add_widget(
            MDBottomNavigationItem(
                name='trading',
                text='Trade',
                icon='swap-horizontal',
                on_tab_press=lambda x: setattr(sm, 'current', 'trading')
            )
        )
        
        bottom_nav.add_widget(
            MDBottomNavigationItem(
                name='agents',
                text='Agents',
                icon='robot',
                on_tab_press=lambda x: setattr(sm, 'current', 'agents')
            )
        )
        
        bottom_nav.add_widget(
            MDBottomNavigationItem(
                name='settings',
                text='Settings',
                icon='cog',
                on_tab_press=lambda x: setattr(sm, 'current', 'settings')
            )
        )
        
        layout.add_widget(bottom_nav)
        
        return layout


def run_mobile_app():
    """Run the mobile application"""
    CryptoTradingMobileApp().run()


if __name__ == '__main__':
    run_mobile_app()
