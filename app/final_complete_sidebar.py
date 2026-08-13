#!/usr/bin/env python3
"""
Final Complete Election Management System with Left Sidebar Navigation
ALL functionality from original gui_app_flet.py without dialogs
"""

# Set matplotlib backend before any other imports
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Flet compatibility

import flet as ft
import sqlite3
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import threading
import time
# from database import VoterDatabase  # Fixed dependency

class FinalCompleteSidebarApp:
    """Final complete election app with left sidebar and ALL original functionality"""
    
    def __init__(self, page: ft.Page):
        self.page = page
        self.current_tab = 11
        # self.db = VoterDatabase()  # Fixed dependency
        self.selected_identity_no = None
        self.selected_records_for_delete = set()
        self.current_edit_order_no = None
        self.setup_page()
        self.create_ui()
        self.load_initial_data()
    
    def setup_page(self):
        """Configure the page"""
        self.page.title = "🗳️ Election Management System - Complete Sidebar"
        self.page.window_width = 1400
        self.page.window_height = 900
        self.page.window_visible = True
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.bgcolor = ft.Colors.WHITE
        self.page.padding = 0
        
        # App bar
        self.page.appbar = ft.AppBar(
            title=ft.Text("🗳️ ระบบจัดการข้อมูลการเลือกตั้ง", color=ft.Colors.WHITE, size=16),
            bgcolor=ft.Colors.BLACK,
            center_title=True,
            toolbar_height=50,
            actions=[
                ft.IconButton(
                    ft.Icons.REFRESH,
                    tooltip="รีเฟรชสถิติ All Data",
                    on_click=self.refresh_all_data,
                    icon_color=ft.Colors.WHITE,
                    icon_size=16
                ),
                ft.IconButton(
                    ft.Icons.DOWNLOAD,
                    tooltip="Export Report",
                    on_click=self.export_report,
                    icon_color=ft.Colors.WHITE,
                    icon_size=16
                )
            ]
        )
    
    def create_ui(self):
        """Create the UI with left sidebar navigation"""
        print("🔧 Creating complete sidebar with ALL functionality...")
        
        # Define navigation items (Dashboard first)
        self.nav_items = [
            {"icon": ft.Icons.DASHBOARD, "text": "📊 แดชบอร์ด", "id": 11},
            {"icon": ft.Icons.MAP, "text": "🌍 แสดงสถานที่เกิด", "id": 0},
            {"icon": ft.Icons.FAMILY_RESTROOM, "text": "👨‍👩‍👧‍👦 จำนวนครอบครัว", "id": 1},
            {"icon": ft.Icons.HOME, "text": "🏠 ข้อมูลบ้าน", "id": 2},
            {"icon": ft.Icons.PLACE, "text": "🌍 ข้อมูลการเกิด", "id": 3},
            {"icon": ft.Icons.BADGE, "text": "🆔 แสดงบัตรประชาชน", "id": 4},
            {"icon": ft.Icons.LOCATION_ON, "text": "🆔 สถานที่เกิด", "id": 5},
            {"icon": ft.Icons.ADD_CIRCLE, "text": "➕ เพิ่มข้อมูล", "id": 6},
            {"icon": ft.Icons.DELETE, "text": "🗑️ ลบข้อมูล", "id": 7},
            {"icon": ft.Icons.EDIT, "text": "✏️ แก้ไขข้อมูล", "id": 8},
            {"icon": ft.Icons.PIE_CHART, "text": "📋 รายงานบัตรประชาชน", "id": 9},
            {"icon": ft.Icons.UPLOAD_FILE, "text": "📄 นำเข้าข้อมูล", "id": 10}
        ]
        
        self.current_tab = 0
        
        # Create beautiful animated sidebar with gradient design
        self.sidebar = ft.Container(
            content=ft.Column([
                # Beautiful gradient header
                ft.Container(
                    content=ft.Column([
                        ft.Text("ระบบย่อย", 
                               size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=5),
                    bgcolor=ft.Colors.BLACK,
                    padding=20,
                    alignment=ft.alignment.center,
                    # Simple header animation
                    animate_opacity=300,
                    border_radius=ft.border_radius.only(bottom_left=15, bottom_right=15),
                    shadow=ft.BoxShadow(
                        spread_radius=0,
                        blur_radius=8,
                        color=ft.Colors.GREY_400,
                        offset=ft.Offset(0, 3)
                    )
                ),
                # Beautiful navigation buttons container
                ft.Container(
                    content=ft.Column([
                        self.create_nav_button(item) for item in self.nav_items
                    ], spacing=4, scroll=ft.ScrollMode.AUTO),
                    padding=ft.padding.all(12),
                    expand=True,
                    bgcolor=ft.Colors.BLACK,
                    # Simple animation for nav buttons
                    animate_opacity=300
                )
            ], spacing=0),
            width=200,  # Slightly wider for better aesthetics
            bgcolor=ft.Colors.BLACK,
            border=ft.border.only(right=ft.BorderSide(1, ft.Colors.GREY_200)),
            border_radius=ft.border_radius.only(top_right=20, bottom_right=20),
            # Simple sidebar animation
            animate_opacity=300,
            # Enhanced shadow with multiple layers
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=15,
                color=ft.Colors.GREY_400,
                offset=ft.Offset(5, 0)
            )
        )
        
        # Create animated content area with simple left-to-right transition
        self.content_area = ft.AnimatedSwitcher(
            content=ft.Container(
                content=self.create_content_for_tab(11),
                expand=True,
                bgcolor=ft.Colors.BROWN_50,
                padding=20,
                key=f"tab_11"  # Unique key for animation
            ),
            transition=ft.AnimatedSwitcherTransition.FADE,
            duration=300,  # Simplified 300ms animation duration
            reverse_duration=300,
            switch_in_curve=ft.AnimationCurve.EASE_OUT,
            switch_out_curve=ft.AnimationCurve.EASE_IN,
            expand=True
        )
        
        # Main layout with sidebar and content
        main_layout = ft.Row([
            self.sidebar,
            self.content_area
        ], expand=True, spacing=0)
        
        self.page.add(main_layout)
        self.page.update()
        
        print("✅ Complete sidebar navigation created successfully!")
    
    def create_nav_button(self, item):
        """Create a beautiful circular navigation button for the sidebar"""
        is_active = item["id"] == self.current_tab
        
        # Define beautiful blue gradient colors for each tab
        colors = [
            ft.Colors.BLUE_400,        # Family - Blue
            ft.Colors.BLUE_500,        # House - Blue  
            ft.Colors.BLUE_600,        # Birth - Blue
            ft.Colors.BLUE_700,        # ID Show - Blue
            ft.Colors.BLUE_800,        # ID Birth Place - Blue
            ft.Colors.INDIGO_400,      # Show Birth Place - Indigo Blue
            ft.Colors.INDIGO_500,      # Insert - Indigo Blue
            ft.Colors.INDIGO_600,      # Delete - Indigo Blue
            ft.Colors.INDIGO_700,      # Update - Indigo Blue
            ft.Colors.INDIGO_800,      # ID Report - Indigo Blue
            ft.Colors.DEEP_PURPLE_400, # Import ID - Deep Purple Blue
        ]
        
        # Get color for this tab (cycle through colors if more tabs than colors)
        tab_color = colors[item["id"] % len(colors)]
        
        # Create animated button with circular design
        button_container = ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Icon(item["icon"], size=18, color=ft.Colors.WHITE),
                    width=35,
                    height=35,
                    bgcolor=tab_color if is_active else ft.Colors.GREY_400,
                    border_radius=25,  # Circular icon container
                    alignment=ft.alignment.center,
                    animate_opacity=300,
                    shadow=ft.BoxShadow(
                        spread_radius=1,
                        blur_radius=6,
                        color=tab_color if is_active else ft.Colors.GREY_300,
                        offset=ft.Offset(0, 3)
                    )
                ),
                ft.Text(
                    item["text"], 
                    size=11, 
                    weight=ft.FontWeight.BOLD if is_active else ft.FontWeight.NORMAL, 
                    color=ft.Colors.WHITE,
                    overflow=ft.TextOverflow.ELLIPSIS,
                    expand=True
                )
            ], spacing=12, alignment=ft.MainAxisAlignment.START),
            bgcolor=ft.Colors.GREY_800 if is_active else ft.Colors.BLUE_900,
            padding=ft.padding.symmetric(horizontal=8, vertical=8),
            border_radius=25,  # Circular button
            margin=ft.margin.symmetric(horizontal=6, vertical=3),
            # Simple animations
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
            animate_opacity=300,
            # Colorful border animation
            border=ft.border.all(2, tab_color if is_active else ft.Colors.TRANSPARENT),
            # Enhanced shadow effect
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color=tab_color if is_active else ft.Colors.TRANSPARENT,
                offset=ft.Offset(0, 4)
            ) if is_active else ft.BoxShadow(
                spread_radius=0,
                blur_radius=4,
                color=ft.Colors.GREY_300,
                offset=ft.Offset(0, 2)
            ),
            on_click=lambda e, tab_id=item["id"]: self.handle_tab_click(tab_id),
            on_hover=lambda e, tab_id=item["id"]: self.handle_button_hover(e, tab_id),
            ink=True,
            ink_color=tab_color
        )
        
        return button_container
    
    def handle_button_hover(self, e, tab_id):
        """Handle beautiful hover effects for circular navigation buttons"""
        colors = [
            ft.Colors.BLUE_400, ft.Colors.BLUE_500, ft.Colors.BLUE_600, ft.Colors.BLUE_700,
            ft.Colors.BLUE_800, ft.Colors.INDIGO_400, ft.Colors.INDIGO_500, ft.Colors.INDIGO_600,
            ft.Colors.INDIGO_700, ft.Colors.INDIGO_800, ft.Colors.DEEP_PURPLE_400
        ]
        tab_color = colors[tab_id % len(colors)]
        is_active = tab_id == self.current_tab
        
        if e.data == "true":  # Mouse enter
            # Simple hover effect - just change background color
            if not is_active:
                e.control.bgcolor = ft.Colors.BLUE_700
        else:  # Mouse leave
            # Restore original styling
            if is_active:
                e.control.bgcolor = ft.Colors.GREY_800
            else:
                e.control.bgcolor = ft.Colors.BLUE_900
        
        e.control.update()
    
    def handle_tab_click(self, tab_id):
        """Handle tab click with simple feedback"""
        # Simple click feedback
        print(f"🎯 Tab {tab_id} clicked")
        
        # Proceed with tab switch
        self.switch_tab(tab_id)
    
    def switch_tab(self, tab_id):
        """Switch to a different tab with simple left-to-right transition"""
        print(f"🔄 Switching to tab {tab_id} with left-to-right transition")
        
        # Add visual feedback during transition
        old_tab = self.current_tab
        self.current_tab = tab_id
        
        # Update sidebar buttons with highlighting
        self.sidebar.content.controls[1].content.controls = [
            self.create_nav_button(item) for item in self.nav_items
        ]
        
        # Create new content with unique key for AnimatedSwitcher
        new_content = ft.Container(
            content=self.create_content_for_tab(tab_id),
            expand=True,
            bgcolor=ft.Colors.BLUE_50,
            padding=20,
            key=f"tab_{tab_id}",  # Unique key triggers animation
            # Simple opacity animation only
            animate_opacity=300
        )
        
        # Always use simple fade transition
        self.content_area.transition = ft.AnimatedSwitcherTransition.FADE
        
        # Simple entrance effect
        new_content.animate_opacity = ft.Animation(300, ft.AnimationCurve.EASE_OUT)
        
        # Update content area with beautiful animation
        self.content_area.content = new_content
        
        # Show subtle feedback
        self.show_snack_bar(f"✨ Switched to {[item['text'] for item in self.nav_items if item['id'] == tab_id][0]}", 
                           ft.Colors.BLUE_600)
        
        self.page.update()
        
        # Auto-load data for specific tabs after page update
        if tab_id == 0:  # Show Birth Place tab
            self.load_all_show_birth_place_records(None)
        elif tab_id == 1:  # Family tab
            self.load_family_data(None)
        elif tab_id == 2:  # House tab
            self.load_house_data(None)
        elif tab_id == 3:  # Birth tab
            self.load_birthplace_data(None)
        elif tab_id == 4:  # Identity tab
            self.load_identity_data(None)
        elif tab_id == 5:  # ID Birth Place tab
            self.load_all_id_birthplace_records(None)
        elif tab_id == 6:  # Insert tab
            self.load_recent_records(None)
        elif tab_id == 7:  # Delete tab
            self.load_all_delete_candidates(None)
        elif tab_id == 8:  # Update tab
            self.load_all_update_candidates(None)
        elif tab_id == 9:  # Identity Report tab
            self.generate_gender_chart(None)
        elif tab_id == 11:  # Dashboard tab
            self.load_dashboard_data(None)
    
    def create_content_for_tab(self, tab_id):
        """Create content for the specified tab"""
        if tab_id == 0:  # Show Birth Place
            return self.create_show_birth_place_tab()
        elif tab_id == 1:  # Family Groups
            return self.create_family_tab()
        elif tab_id == 2:  # House Analysis
            return self.create_house_tab()
        elif tab_id == 3:  # Birth Places
            return self.create_birthplace_tab()
        elif tab_id == 4:  # ID Numbers
            return self.create_identity_number_tab()
        elif tab_id == 5:  # ID Birth Place
            return self.create_id_birthplace_tab()
        elif tab_id == 6:  # Insert Data
            return self.create_insert_election_tab()
        elif tab_id == 7:  # Delete Data
            return self.create_delete_election_tab()
        elif tab_id == 8:  # Update Data
            return self.create_update_election_tab()
        elif tab_id == 9:  # Identity Report
            return self.create_identity_report_tab()
        elif tab_id == 10:  # Import Excel Identity Card
            return self.create_import_identity_tab()
        elif tab_id == 11:  # Dashboard
            return self.create_dashboard_tab()
    
    # EXACT COPY of create_identity_number_tab with all functionality
    def create_identity_number_tab(self):
        """Create identity number tab with complete functionality - NO DIALOGS"""
        
        # Create identity table with ALL fields from election_c
        self.identity_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Order", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Identity No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Title", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Address No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("หมู่ที่", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Day Birth", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Month", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Year", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Gender", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Remark", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Birthday", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Create identity details area
        self.identity_details = ft.Column([
            ft.Text("👆 คลิกบนข้อมูลบัตรประชาชนด้านบนเพื่อดูรายละเอียด...", 
                   size=16, color=ft.Colors.GREY_600)
        ])
        
        # Direct search input field (NO DIALOG)
        self.search_input_field = ft.TextField(
            label="🔍 Search by Identity Number",
            hint_text="Enter ID number and press Enter or click Search...",
            expand=True,
            on_submit=self.direct_search_identity
        )
        
        return ft.Column([
            ft.Text("🆔 ระบบจัดการบัตรประชาชน", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK),
            
            # Controls card
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🎛️ การควบคุม", weight=ft.FontWeight.BOLD, size=16),
                        ft.Text("📋 ข้อมูลบัตรประชาชนจะโหลดอัตโนมัติเมื่อคลิกแท็บนี้", 
                               size=12, color=ft.Colors.GREY_600),
                        ft.Row([
                            ft.ElevatedButton("🗑️ ลบข้อมูล N/A", 
                                            bgcolor=ft.Colors.RED_600, 
                                            color=ft.Colors.WHITE,
                                            on_click=self.delete_empty_records)
                        ], spacing=10)
                    ], spacing=10),
                    padding=15
                ),
                elevation=2
            ),
            
            # Direct search card (NO DIALOG)
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🔍 ค้นหาโดยตรง", weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.BLUE_700),
                        ft.Row([
                            self.search_input_field,
                            ft.ElevatedButton("🔍 ค้นหาทันที", 
                                            bgcolor=ft.Colors.BLUE_600, 
                                            color=ft.Colors.WHITE,
                                            on_click=self.direct_search_identity)
                        ], spacing=10)
                    ], spacing=10),
                    padding=15,
                    bgcolor=ft.Colors.BLUE_50
                ),
                elevation=2
            ),
            
            # Data table card with horizontal and vertical scrolling
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📋 ฟิลด์ทั้งหมดจากตาราง election_c", weight=ft.FontWeight.BOLD),
                        ft.Text("👆 เลื่อนแนวนอนและแนวตั้งเพื่อดูข้อมูลทั้งหมด", 
                               size=12, color=ft.Colors.BLUE_600),
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.identity_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=400,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10,
                            bgcolor=ft.Colors.WHITE
                        )
                    ]),
                    padding=20
                ),
                elevation=4
            ),
            
            # Details card
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 รายละเอียดบัตรประชาชน", weight=ft.FontWeight.BOLD),
                        self.identity_details
                    ]),
                    padding=20
                ),
                elevation=4
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    # EXACT COPY of all database functions from original
    def load_identity_data(self, e=None):
        """Load identity number data from demo_voters.db election_c table"""
        try:
            # Connect to demo_voters.db database
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Load ALL data from election_c table
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                ORDER BY order_no
                LIMIT 100
            """)
            identity_data = cursor.fetchall()
            
            conn.close()
            
            # Clear existing rows
            self.identity_table.rows.clear()
            
            # Add new rows with ALL fields
            for identity in identity_data:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, gender, remark, birthday = identity
                self.identity_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(title or ""))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(address_no or ""))),
                            ft.DataCell(ft.Text(str(mo_address or ""))),
                            ft.DataCell(ft.Text(str(day_birth or ""))),
                            ft.DataCell(ft.Text(str(month or ""))),
                            ft.DataCell(ft.Text(str(year or ""))),
                            ft.DataCell(ft.Text(str(gender or ""))),
                            ft.DataCell(ft.Text(str(remark or ""))),
                            ft.DataCell(ft.Text(str(birthday or "")))
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_identity_select(id_num)
                    )
                )
            
            self.page.update()
            
            # Show message about loaded data
            if identity_data:
                self.show_snack_bar(f"🆔 Loaded {len(identity_data)} identity records from election_c", ft.Colors.GREEN_600)
            else:
                self.show_snack_bar("📋 No identity data found in election_c table", ft.Colors.ORANGE_600)
            
        except Exception as e:
            self.show_snack_bar(f"Error loading identity data: {e}", ft.Colors.RED_600)
    
    def on_identity_select(self, identity_no):
        """Handle identity selection (similar to on_birthplace_select)"""
        try:
            # Store selected identity for delete operation
            self.selected_identity_no = identity_no
            
            # Connect to demo_voters.db database
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get full details for selected identity
            cursor.execute("""
                SELECT * FROM election_c WHERE identity_no = ?
            """, (identity_no,))
            identity = cursor.fetchone()
            
            if identity:
                # Get column names
                cursor.execute("PRAGMA table_info(election_c)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Format details for modal dialog
                details_text = f"🆔 Identity Details for: {identity_no}\n\n"
                details_text += "=" * 50 + "\n\n"
                
                # Add all field details
                for i, value in enumerate(identity):
                    if i < len(columns):
                        field_name = columns[i]
                        display_value = str(value) if value is not None else "N/A"
                        details_text += f"{field_name}: {display_value}\n"
                
                # Show identity details in modal dialog instead of panel
                self.show_desktop_modal_dialog(f"🆔 รายละเอียดบัตรประชาชน: {identity_no}", details_text, 500, 600)
            else:
                self.show_snack_bar(f"❌ No details found for identity: {identity_no}", ft.Colors.RED_600)
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading identity details: {e}")
            self.show_snack_bar(f"❌ Error loading details: {e}", ft.Colors.RED_600)
    
    def direct_search_identity(self, e):
        """Direct search identity (NO DIALOG)"""
        search_term = self.search_input_field.value.strip()
        if not search_term:
            self.show_snack_bar("⚠️ Please enter an ID number to search", ft.Colors.ORANGE_600)
            return
        
        try:
            # Connect to demo_voters.db database
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Search for matching records - ALL fields
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no LIKE ? OR name LIKE ? OR surname LIKE ?
                ORDER BY order_no
            """, (f"%{search_term}%", f"%{search_term}%", f"%{search_term}%"))
            
            identity_data = cursor.fetchall()
            conn.close()
            
            # Clear existing rows
            self.identity_table.rows.clear()
            
            # Add matching rows with ALL fields
            for identity in identity_data:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, gender, remark, birthday = identity
                self.identity_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(title or ""))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(address_no or ""))),
                            ft.DataCell(ft.Text(str(mo_address or ""))),
                            ft.DataCell(ft.Text(str(day_birth or ""))),
                            ft.DataCell(ft.Text(str(month or ""))),
                            ft.DataCell(ft.Text(str(year or ""))),
                            ft.DataCell(ft.Text(str(gender or ""))),
                            ft.DataCell(ft.Text(str(remark or ""))),
                            ft.DataCell(ft.Text(str(birthday or "")))
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_identity_select(id_num)
                    )
                )
            
            self.page.update()
            
            # Show search results
            if identity_data:
                self.show_snack_bar(f"🔍 Found {len(identity_data)} records matching '{search_term}'", ft.Colors.BLUE_600)
            else:
                self.show_snack_bar(f"🔍 No records found matching '{search_term}'", ft.Colors.ORANGE_600)
            
        except Exception as e:
            self.show_snack_bar(f"❌ Search error: {e}", ft.Colors.RED_600)
    
    def delete_empty_records(self, e):
        """Delete empty records from election_c table"""
        try:
            # Connect to demo_voters.db database
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Count empty records first
            cursor.execute("""
                SELECT COUNT(id) FROM election_c 
                WHERE (identity_no IS NULL OR identity_no = '' OR identity_no = 'N/A')
                   OR (name IS NULL OR name = '' OR name = 'N/A')
                   OR (surname IS NULL OR surname = '' OR surname = 'N/A')
            """)
            empty_count = cursor.fetchone()[0]
            
            if empty_count > 0:
                # Delete empty records
                cursor.execute("""
                    DELETE FROM election_c 
                    WHERE (identity_no IS NULL OR identity_no = '' OR identity_no = 'N/A')
                       OR (name IS NULL OR name = '' OR name = 'N/A')
                       OR (surname IS NULL OR surname = '' OR surname = 'N/A')
                """)
                
                conn.commit()
                deleted_count = cursor.rowcount
                
                self.show_snack_bar(f"🗑️ Deleted {deleted_count} empty records", ft.Colors.GREEN_600)
                
                # รีเฟรชสถิติ the identity table
                self.load_identity_data()
            else:
                self.show_snack_bar("✅ No empty records found to delete", ft.Colors.BLUE_600)
            
            conn.close()
            
        except Exception as e:
            self.show_snack_bar(f"❌ Error deleting empty records: {e}", ft.Colors.RED_600)
    
    def create_import_tab(self):
        """Create the import data tab - EXACT COPY from original"""
        # File picker
        self.file_picker = ft.FilePicker(
            on_result=self.on_file_picked
        )
        self.page.overlay.append(self.file_picker)
        
        # File path display
        self.file_path_text = ft.Text("ยังไม่ได้เลือกไฟล์", color=ft.Colors.GREY_600)
        
        # Import status
        self.import_status = ft.Text("พร้อมนำเข้าข้อมูล...", size=14)
        
        # Progress bar
        self.progress_bar = ft.ProgressBar(visible=False)
        
        # Database statistics
        self.stats_text = ft.Text("Loading statistics...", size=12)
        
        return ft.Column([
            # Import section
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📥 นำเข้าไฟล์ Excel", size=18, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Row([
                            ft.ElevatedButton(
                                "📁 เลือกไฟล์ Excel",
                                icon=ft.Icons.FOLDER_OPEN,
                                on_click=lambda _: self.file_picker.pick_files(
                                    dialog_title="Select Excel File",
                                    file_type=ft.FilePickerFileType.CUSTOM,
                                    allowed_extensions=["xlsx", "xls"]
                                )
                            ),
                            self.file_path_text
                        ]),
                        ft.ElevatedButton(
                            "🚀 นำเข้าข้อมูล",
                            bgcolor=ft.Colors.BLUE_600,
                            color=ft.Colors.WHITE,
                            on_click=self.import_data
                        ),
                        self.progress_bar,
                        self.import_status
                    ]),
                    padding=20
                )
            ),
            
            # Statistics section
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📊 สถิติฐานข้อมูล", size=18, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.stats_text,
                        ft.ElevatedButton(
                            "🔄 รีเฟรชสถิติ",
                            bgcolor=ft.Colors.GREEN_600,
                            color=ft.Colors.WHITE,
                            on_click=self.update_stats
                        )
                    ]),
                    padding=20
                )
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    def create_family_tab(self):
        """Create the family analysis tab - EXACT COPY from original"""
        
        # Family voter database table (joined with surname table) - with Details button
        self.family_voter_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Total Votes", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Houses", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Note1", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Note2", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Note3", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Details", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=60
        )
        
        # Family details
        self.family_details = ft.Text(
            "เลือกครอบครัวเพื่อดูรายละเอียด...", 
            size=12, 
            selectable=True,
            text_align=ft.TextAlign.LEFT
        )
        
        return ft.Column([
            # Controls
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🎛️ การควบคุม", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Text("👨‍👩‍👧‍👦 ครอบครัวจะแสดงข้อมูลโดยอัตโนมัติจากฐานข้อมูล", size=12, color=ft.Colors.BLUE_700)
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.YELLOW
                )
            ),
            
            # Family table
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("👨‍👩‍👧‍👦 กลุ่มครอบครัวพร้อมหมายเหตุ", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Column([
                                self.family_voter_table
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=500,
                            width=1200,
                            bgcolor=ft.Colors.WHITE
                        )
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.GREEN
                )
            ),
            
            # Family details
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🔍 รายละเอียดครอบครัว", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Container(
                            content=self.family_details,
                            padding=10,
                            bgcolor=ft.Colors.GREY_50,
                            border_radius=5,
                            height=300,
                            width=1200
                        )
                    ]),
                    padding=20
                )
            ),
            

        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    def create_house_tab(self):
        """Create the house analysis tab - EXACT COPY from original"""
        # House controls
        self.house_min_voters = ft.TextField(label="Min Voters", value="2", width=100)
        
        # House table (now using address from election_c)
        self.house_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Address No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("หมู่ที่", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("People Count", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Family Names", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Support Level", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Details", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=60
        )
        
        # House details
        self.house_details = ft.Text("เลือกที่อยู่เพื่อดูรายละเอียด...", size=12)
        
        return ft.Column([
            # Controls
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🎛️ การควบคุม", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Row([
                            self.house_min_voters
                        ], spacing=10)
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.YELLOW
                )
            ),
            
            # House table
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🏠 วิเคราะห์ข้อมูลบ้าน", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Column([
                                self.house_table
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=300,
                            bgcolor=ft.Colors.WHITE
                        )
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.GREEN
                )
            ),
            
            # House details
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🔍 รายละเอียดที่อยู่", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.house_details
                    ]),
                    padding=20
                )
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    def create_birthplace_tab(self):
        """Create the birthplace analysis tab - EXACT COPY from original"""
        # Birthplace controls (automatic loading)
        
        # Birthplace table (now using election_c + id_birthplace)
        self.birthplace_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Birth Code", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("District", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Province", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("People Count", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Support Level", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Details", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=60
        )
        
        # Birthplace details
        self.birthplace_details = ft.Text("เลือกรหัสสถานที่เกิดเพื่อดูรายละเอียด...", size=12)
        
        return ft.Column([
            # Birthplace table
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🌍 วิเคราะห์รหัสสถานที่เกิด", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Column([
                                self.birthplace_table
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=300,
                            bgcolor=ft.Colors.WHITE
                        )
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.GREEN
                )
            ),
            
            # Birthplace details
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🔍 รายละเอียดรหัสสถานที่เกิด", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.birthplace_details
                    ]),
                    padding=20
                )
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    def create_campaign_tab(self):
        """Create the campaign activities tab - EXACT COPY from original"""
        # Form elements
        self.activity_type = ft.Dropdown(
            label="Activity Type",
            options=[
                ft.dropdown.Option("Family Visit"),
                ft.dropdown.Option("House Visit"),
                ft.dropdown.Option("Phone Call"),
                ft.dropdown.Option("Regional Meeting"),
                ft.dropdown.Option("Event")
            ],
            width=200
        )
        
        self.target_group = ft.Dropdown(
            label="Target Group",
            options=[
                ft.dropdown.Option("Family"),
                ft.dropdown.Option("House"),
                ft.dropdown.Option("Birthplace")
            ],
            width=150
        )
        
        self.target_id = ft.TextField(label="Target ID", width=150)
        self.activity_description = ft.TextField(label="Description", expand=True)
        
        # Activities table
        self.activities_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Date")),
                ft.DataColumn(ft.Text("Type")),
                ft.DataColumn(ft.Text("Target")),
                ft.DataColumn(ft.Text("Target ID")),
                ft.DataColumn(ft.Text("Description")),
                ft.DataColumn(ft.Text("Result"))
            ],
            rows=[],
            data_row_max_height=60
        )
        
        return ft.Column([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 เพิ่มกิจกรรมการรณรงค์", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Row([
                            self.activity_type,
                            self.target_group,
                            self.target_id
                        ], spacing=10),
                        self.activity_description,
                        ft.ElevatedButton(
                            "➕ Add Activity",
                            bgcolor=ft.Colors.BLACK,
                            color=ft.Colors.WHITE,
                            on_click=self.add_campaign_activity
                        )
                    ]),
                    padding=15
                )
            ),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📋 กิจกรรมล่าสุด", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Column([
                                self.activities_table
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=300,
                            bgcolor=ft.Colors.WHITE
                        )
                    ]),
                    padding=15
                )
            )
        ], spacing=10, scroll=ft.ScrollMode.AUTO)
    
    def create_reports_tab(self):
        """Create the reports and charts tab with working matplotlib charts"""
        # Chart display container
        self.chart_container = ft.Container(
            content=ft.Text("📊 คลิกปุ่มด้านบนเพื่อสร้างกราฟ", 
                           size=14, color=ft.Colors.GREY_600),
            height=500,
            bgcolor=ft.Colors.WHITE,
            alignment=ft.alignment.center,
            border=ft.border.all(2, ft.Colors.GREY_300),
            border_radius=8
        )
        
        return ft.Column([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📊 รายงานและกราฟ", size=18, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        ft.Row([
                            ft.ElevatedButton(
                                "👨‍👩‍👧‍👦 กราฟครอบครัว",
                                bgcolor=ft.Colors.BLUE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_family_chart
                            ),
                            ft.ElevatedButton(
                                "🏠 กราฟบ้าน", 
                                bgcolor=ft.Colors.GREEN_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_house_chart
                            ),
                            ft.ElevatedButton(
                                "📊 สรุปสถิติ",
                                bgcolor=ft.Colors.PURPLE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_stats_chart
                            ),
                            ft.ElevatedButton(
                                "📄 ส่งออกรายงาน",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.export_report
                            )
                        ], spacing=10)
                    ]),
                    padding=20
                )
            ),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📈 พื้นที่แสดงกราฟ", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.chart_container
                    ]),
                    padding=20
                )
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    def create_identity_report_tab(self):
        """Create the identity card reports and charts tab with working matplotlib charts from election_c table"""
        # Chart display container
        self.id_chart_container = ft.Container(
            content=ft.Text("📊 คลิกปุ่มด้านบนเพื่อสร้างกราฟบัตรประชาชน", 
                           size=14, color=ft.Colors.GREY_600),
            height=500,
            bgcolor=ft.Colors.WHITE,
            alignment=ft.alignment.center,
            border=ft.border.all(2, ft.Colors.GREY_300),
            border_radius=8
        )
        
        return ft.Column([
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📋 รายงานและกราฟบัตรประชาชน", size=18, weight=ft.FontWeight.BOLD),
                        ft.Text("📊 การวิเคราะห์จากตาราง election_c", size=14, color=ft.Colors.PINK_600),
                        ft.Divider(),
                        ft.Row([
                            ft.ElevatedButton(
                                "👥 การกระจายเพศ",
                                bgcolor=ft.Colors.PINK_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_gender_chart
                            ),
                            ft.ElevatedButton(
                                "📅 การวิเคราะห์ปีเกิด", 
                                bgcolor=ft.Colors.PURPLE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_birth_year_chart
                            ),
                            ft.ElevatedButton(
                                "🏘️ การกระจายที่อยู่",
                                bgcolor=ft.Colors.INDIGO_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_address_chart
                            ),
                            ft.ElevatedButton(
                                "📈 สถิติบัตรประชาชน",
                                bgcolor=ft.Colors.DEEP_PURPLE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_identity_stats_chart
                            )
                        ], spacing=10),
                        ft.Row([
                            ft.ElevatedButton(
                                "🎂 การเกิดรายเดือน",
                                bgcolor=ft.Colors.TEAL_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_monthly_births_chart
                            ),
                            ft.ElevatedButton(
                                "📊 การกระจายคำนำหน้า",
                                bgcolor=ft.Colors.GREEN_600,
                                color=ft.Colors.WHITE,
                                on_click=self.generate_title_chart
                            ),
                            ft.ElevatedButton(
                                "📄 ส่งออกรายงานบัตรประชาชน",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.export_identity_report
                            )
                        ], spacing=10)
                    ]),
                    padding=20
                )
            ),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📈 พื้นที่แสดงกราฟบัตรประชาชน", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.id_chart_container
                    ]),
                    padding=20
                )
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)
    
    def create_import_identity_tab(self):
        """Create the import Excel identity card tab with format validation and preview"""
        # File picker for Excel/CSV files
        self.identity_file_picker = ft.FilePicker(
            on_result=self.on_identity_file_picked
        )
        self.page.overlay.append(self.identity_file_picker)
        
        # File path display
        self.identity_file_path_text = ft.Text("ยังไม่ได้เลือกไฟล์", color=ft.Colors.GREY_600, size=14)
        
        # Import status
        self.identity_import_status = ft.Text("พร้อมนำเข้าข้อมูลบัตรประชาชน...", size=14)
        
        # Progress bar
        self.identity_progress_bar = ft.ProgressBar(visible=False)
        
        # Preview table for validation
        self.identity_preview_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Order", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Identity No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Title", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Address No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("หมู่ที่", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Day Birth", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Month", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Year", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Gender", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Remark", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Birthday", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=50,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Current data table showing records from election_c
        self.current_identity_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Order", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Identity No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Title", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Address No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("หมู่ที่", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Day Birth", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Month", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Year", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Gender", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Remark", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Birthday", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=50,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        return ft.Column([
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📄 นำเข้าข้อมูลบัตรประชาชนจาก Excel", size=18, weight=ft.FontWeight.BOLD),
                        ft.Text("📋 นำเข้าไฟล์ Excel/CSV ที่ตรงกับรูปแบบตาราง election_c", size=12, color=ft.Colors.BLUE_600),
                        ft.Divider(),
                        
                        # Expected format info
                        ft.Container(
                            content=ft.Column([
                                ft.Text("📋 รูปแบบ Excel/CSV ที่คาดหวัง:", size=14, weight=ft.FontWeight.BOLD),
                                ft.Text("Columns: order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday", 
                                       size=11, color=ft.Colors.GREY_700),
                                ft.Text("💡 Tip: First row should contain column headers", size=11, color=ft.Colors.GREEN_600)
                            ]),
                            bgcolor=ft.Colors.GREY_100,
                            padding=10,
                            border_radius=5
                        ),
                        
                        # File selection
                        ft.Row([
                            ft.ElevatedButton(
                                "Browse Excel/CSV File",
                                icon=ft.Icons.FOLDER_OPEN,
                                bgcolor=ft.Colors.BLUE_600,
                                color=ft.Colors.WHITE,
                                on_click=lambda _: self.identity_file_picker.pick_files(
                                    dialog_title="Select Excel or CSV File",
                                    file_type=ft.FilePickerFileType.CUSTOM,
                                    allowed_extensions=["xlsx", "xls", "csv"]
                                )
                            ),
                            self.identity_file_path_text
                        ], spacing=10),
                        
                        # Action buttons
                        ft.Row([
                            ft.ElevatedButton(
                                "🔍 Preview & Validate",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.preview_identity_file
                            ),
                            ft.ElevatedButton(
                                "🚀 นำข้อมูลเข้า",
                                bgcolor=ft.Colors.GREEN_600,
                                color=ft.Colors.WHITE,
                                on_click=self.import_identity_data
                            ),
                            ft.ElevatedButton(
                                "🔄 รีเฟรชสถิติ Current Data",
                                bgcolor=ft.Colors.PURPLE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.load_current_identity_data
                            )
                        ], spacing=10),
                        
                        self.identity_progress_bar,
                        self.identity_import_status
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.WHITE
                )
            ),
            
            # Preview section
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("👁️ ตัวอย่างไฟล์และการตรวจสอบ", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("Preview first 10 rows to validate format before importing", size=12, color=ft.Colors.ORANGE_600),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.identity_preview_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=250,
                            bgcolor=ft.Colors.WHITE,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10
                        )
                    ]),
                    padding=15,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Current data section
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("💾 ข้อมูลฐานข้อมูลปัจจุบัน (20 รายการล่าสุด)", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("ข้อมูลสดจากตาราง election_c", size=12, color=ft.Colors.PURPLE_600),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.current_identity_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=300,
                            bgcolor=ft.Colors.WHITE,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10
                        )
                    ]),
                    padding=15,
                    bgcolor=ft.Colors.GREY_100
                )
            )
        ], spacing=20, scroll=ft.ScrollMode.AUTO)

    def create_dashboard_tab(self):
        """Create a dashboard showing quick-access buttons for all tabs (no charts)"""
        # Button rows linking to each tab
        def jump_button(icon, text, tab_id, color):
            return ft.ElevatedButton(
                text,
                icon=icon,
                bgcolor=color,
                color=ft.Colors.WHITE,
                on_click=lambda _: self.switch_tab(tab_id)
            )
        
        header = ft.Text("📊 แดชบอร์ด: ทางลัดไปยังเมนูย่อยทั้งหมด", size=20, weight=ft.FontWeight.BOLD)
        
        return ft.Column([
            header,
            ft.Divider(),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("เมนูข้อมูลหลัก", size=16, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            jump_button(ft.Icons.MAP, "🌍 แสดงสถานที่เกิด", 0, ft.Colors.INDIGO_500),
                            jump_button(ft.Icons.FAMILY_RESTROOM, "👨‍👩‍👧‍👦 จำนวนครอบครัว", 1, ft.Colors.BLUE_500),
                            jump_button(ft.Icons.HOME, "🏠 ข้อมูลบ้าน", 2, ft.Colors.CYAN_600),
                            jump_button(ft.Icons.PLACE, "🌍 ข้อมูลการเกิด", 3, ft.Colors.TEAL_600),
                        ], wrap=True, spacing=10)
                    ], spacing=10),
                    padding=16
                )
            ),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("บัตรประชาชนและรายงาน", size=16, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            jump_button(ft.Icons.BADGE, "🆔 แสดงบัตรประชาชน", 4, ft.Colors.DEEP_PURPLE_500),
                            jump_button(ft.Icons.LOCATION_ON, "🆔 สถานที่เกิด", 5, ft.Colors.PURPLE_500),
                            jump_button(ft.Icons.PIE_CHART, "📋 รายงานบัตรประชาชน", 9, ft.Colors.PINK_600),
                        ], wrap=True, spacing=10)
                    ], spacing=10),
                    padding=16
                )
            ),
            ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("การจัดการข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                        ft.Row([
                            jump_button(ft.Icons.ADD_CIRCLE, "➕ เพิ่มข้อมูล", 6, ft.Colors.GREEN_600),
                            jump_button(ft.Icons.DELETE, "🗑️ ลบข้อมูล", 7, ft.Colors.RED_600),
                            jump_button(ft.Icons.EDIT, "✏️ แก้ไขข้อมูล", 8, ft.Colors.ORANGE_600),
                            jump_button(ft.Icons.UPLOAD_FILE, "📄 นำเข้าข้อมูล", 10, ft.Colors.BROWN_600),
                        ], wrap=True, spacing=10)
                    ], spacing=10),
                    padding=16
                )
            ),
        ], spacing=16, scroll=ft.ScrollMode.AUTO)

    def load_dashboard_data(self, e):
        """Load key metrics from election_c and render mini charts"""
        try:
            import sqlite3
            import matplotlib.pyplot as plt
            import io, base64
            
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Total voters
            cursor.execute("SELECT COUNT(*) FROM election_c")
            total_voters = cursor.fetchone()[0]
            
            # Total families (distinct surname, excluding empty)
            cursor.execute("""
                SELECT COUNT(DISTINCT surname)
                FROM election_c
                WHERE surname IS NOT NULL AND TRIM(surname) != '' AND surname != 'N/A'
            """)
            total_families = cursor.fetchone()[0]
            
            # Total houses (distinct no/mo combination)
            cursor.execute("""
                SELECT COUNT(DISTINCT no_address || '|' || mo_address)
                FROM election_c
                WHERE no_address IS NOT NULL AND TRIM(no_address) != ''
                  AND mo_address IS NOT NULL AND TRIM(mo_address) != ''
            """)
            total_houses = cursor.fetchone()[0]
            
            # Birthplaces count proxy (distinct month/day combo)
            cursor.execute("""
                SELECT COUNT(*) FROM (
                    SELECT DISTINCT day_birth, month FROM election_c
                    WHERE (day_birth IS NOT NULL AND TRIM(day_birth) != '')
                       OR (month IS NOT NULL AND TRIM(month) != '')
                )
            """)
            total_birth_places = cursor.fetchone()[0]
            
            conn.close()
            
            # Update stats text (Thai labels)
            self.dashboard_stats_text.value = (
                f"👥 จำนวนผู้มีสิทธิ์เลือกตั้งทั้งหมด: {total_voters:,}\n"
                f"👨‍👩‍👧‍👦 จำนวนครอบครัว (นามสกุล): {total_families:,}\n"
                f"🏠 จำนวนบ้าน: {total_houses:,}\n"
                f"🌍 จำนวนสถานที่เกิด (โดยประมาณ): {total_birth_places:,}"
            )
            self.dashboard_stats_text.update()
            
            # Mini chart: top 5 surnames by voters
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            cursor.execute("""
                SELECT surname, COUNT(*) as c
                FROM election_c
                WHERE surname IS NOT NULL AND TRIM(surname) != '' AND surname != 'N/A'
                GROUP BY surname
                ORDER BY c DESC
                LIMIT 5
            """)
            top_families = cursor.fetchall()
            conn.close()
            
            labels = [r[0] for r in top_families]
            values = [r[1] for r in top_families]
            
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.figure(figsize=(5, 3))
            plt.bar(labels, values, color='#1976D2')
            plt.title('นามสกุลสูงสุด 5 อันดับ')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            img_base64_1 = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            
            self.dashboard_chart_container.controls = [
                ft.Card(content=ft.Container(content=ft.Image(src_base64=img_base64_1, width=360, height=220, fit=ft.ImageFit.CONTAIN), padding=12))
            ]
            self.dashboard_chart_container.update()
            
            self.show_snack_bar("✅ โหลดแดชบอร์ดสำเร็จ", ft.Colors.GREEN_600)
        except Exception as ex:
            self.show_snack_bar(f"❌ โหลดแดชบอร์ดผิดพลาด: {ex}", ft.Colors.RED_600)
    
    def create_id_birthplace_tab(self):
        """Create ID Birth Place tab showing identity numbers and birthplace from election_c table"""
        
        # Create ID birthplace table with birthplace codes and locations
        self.id_birthplace_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Birth Code", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("District", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Province", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Additional Info", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Details", weight=ft.FontWeight.BOLD)),
            ],
            rows=[],
            data_row_max_height=50,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Search field
        self.id_birthplace_search_field = ft.TextField(
            label="🔍 Search by Code, District, or Province",
            hint_text="Enter birth place code, district name, or province name",
            width=400,
            on_change=self.search_id_birthplace_records
        )
        
        # Details panel for selected record
        self.id_birthplace_details = ft.Column([
            ft.Text("👆 Click on any record above to see details...", 
                   size=16, color=ft.Colors.GREY_600)
        ])
        
        return ft.Column([
            # Header card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.LOCATION_ON, size=30, color=ft.Colors.BLUE_600),
                        ft.Text("🆔 ข้อมูลสถานที่เกิด", 
                               size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
                        ft.Icon(ft.Icons.PLACE, size=30, color=ft.Colors.BLUE_600),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=15,
                    bgcolor=ft.Colors.BLUE_100
                )
            ),
            
            # Search card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                                                        ft.Text("🔍 ค้นหารหัสสถานที่เกิดและตำแหน่ง", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        
                        ft.Row([
                            self.id_birthplace_search_field
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        
                        ft.Row([
                            ft.ElevatedButton(
                                "📊 สถิติ",
                                bgcolor=ft.Colors.GREEN_600,
                                color=ft.Colors.WHITE,
                                on_click=self.show_id_birthplace_stats,
                                icon=ft.Icons.ANALYTICS
                            ),
                            ft.ElevatedButton(
                                "📄 Export Data",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.export_id_birthplace_data,
                                icon=ft.Icons.DOWNLOAD
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_AROUND)
                    ], spacing=15),
                    padding=20,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Data table card with scrollbars
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                                                        ft.Text("📋 รหัสสถานที่เกิดและตำแหน่ง", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("👆 เลื่อนแนวนอนและแนวตั้งเพื่อดูข้อมูลทั้งหมด | คลิกบนแถวเพื่อดูรายละเอียด", 
                               size=12, color=ft.Colors.BLUE_600),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.id_birthplace_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=400,
                            bgcolor=ft.Colors.WHITE,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10
                        )
                    ]),
                    padding=15,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Details card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 รายละเอียดข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.id_birthplace_details
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.BLUE_50
                )
            )
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
    
    def create_insert_election_tab(self):
        """Create tab for inserting data into election_c table - EXACT COPY from original"""
        
        # Form fields for election_c table
        self.insert_identity_no = ft.TextField(
            label="เลขประจำตัวประชาชน",
            hint_text="e.g., 3-6405-00108-25-0",
            width=300
        )
        
        self.insert_name_title = ft.Dropdown(
            label="คำนำหน้า",
            options=[
                ft.dropdown.Option("นาย"),
                ft.dropdown.Option("นาง"),
                ft.dropdown.Option("น.ส."),
                ft.dropdown.Option("เด็กชาย"),
                ft.dropdown.Option("เด็กหญิง")
            ],
            width=120
        )
        
        self.insert_name = ft.TextField(label="ชื่อ", width=200)
        self.insert_surname = ft.TextField(label="นามสกุล", width=200)
        self.insert_no_address = ft.TextField(label="เลขที่บ้าน", width=100)
        self.insert_mo_address = ft.TextField(label="หมู่ที่", width=100, value="1")
        
        self.insert_day_birth = ft.TextField(label="Day", width=80)
        self.insert_month = ft.Dropdown(
            label="เดือน",
            options=[
                ft.dropdown.Option("ม.ค."), ft.dropdown.Option("ก.พ."), ft.dropdown.Option("มี.ค."),
                ft.dropdown.Option("เม.ย."), ft.dropdown.Option("พ.ค."), ft.dropdown.Option("มิ.ย."),
                ft.dropdown.Option("ก.ค."), ft.dropdown.Option("ส.ค."), ft.dropdown.Option("ก.ย."),
                ft.dropdown.Option("ต.ค."), ft.dropdown.Option("พ.ย."), ft.dropdown.Option("ธ.ค.")
            ],
            width=100
        )
        self.insert_year = ft.TextField(label="Year (พ.ศ.)", width=100)
        
        self.insert_sex = ft.Dropdown(
            label="เพศ",
            options=[
                ft.dropdown.Option("ชาย"),
                ft.dropdown.Option("หญิง")
            ],
            width=100
        )
        
        self.insert_remark = ft.TextField(label="หมายเหตุ (ไม่บังคับ)", width=300)
        
        # Recent records table with 5 key fields from election_c - Now editable
        self.recent_records_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("ลำดับ", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("เลขประจำตัว", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("ชื่อ", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("เพศ", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=50,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Store editable data for the table
        self.editable_table_data = []
        
        # Insert tab details panel (like ID Show tab)
        self.insert_details = ft.Column([
            ft.Text("👆 คลิกบนข้อมูลใดๆ เพื่อดูรายละเอียด...", 
                   size=16, color=ft.Colors.GREY_600)
        ])
        
        return ft.Column([
            # Header card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PERSON_ADD, size=30, color=ft.Colors.GREEN_600),
                        ft.Text("📝 เพิ่มข้อมูลการเลือกตั้งใหม่", 
                               size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_800),
                        ft.Icon(ft.Icons.SAVE, size=30, color=ft.Colors.GREEN_600),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=15,
                    bgcolor=ft.Colors.GREEN_100
                )
            ),
            
            # Form card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 เพิ่มข้อมูลการเลือกตั้งใหม่", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        
                        # Identity and basic info
                        ft.Row([
                            self.insert_identity_no,
                            self.insert_name_title,
                        ], alignment=ft.MainAxisAlignment.START),
                        
                        # Name fields
                        ft.Row([
                            self.insert_name,
                            self.insert_surname,
                            self.insert_sex
                        ]),
                        
                        # Address fields
                        ft.Row([
                            self.insert_no_address,
                            self.insert_mo_address,
                        ]),
                        
                        # Birth date fields
                        ft.Row([
                            self.insert_day_birth,
                            self.insert_month,
                            self.insert_year
                        ]),
                        
                        # Remark
                        ft.Row([
                            self.insert_remark
                        ]),
                        
                        # Action buttons
                        ft.Row([
                            ft.ElevatedButton(
                                "💾 บันทึกข้อมูล",
                                bgcolor=ft.Colors.GREEN_600,
                                color=ft.Colors.WHITE,
                                on_click=self.save_election_record,
                                icon=ft.Icons.SAVE
                            ),
                            ft.ElevatedButton(
                                "🔄 ล้างฟอร์ม",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.clear_election_form,
                                icon=ft.Icons.CLEAR
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_AROUND)
                    ], spacing=15),
                    padding=20,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Recent records card with horizontal and vertical scrolling
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📋 5 ฟิลด์หลักจากข้อมูลที่เพิ่มล่าสุด", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("👆 คลิกบนข้อมูลใดๆ เพื่อดูรายละเอียดเต็ม", 
                               size=12, color=ft.Colors.GREEN_600),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.recent_records_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=250,
                            bgcolor=ft.Colors.WHITE,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10
                        )
                    ]),
                    padding=15,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Details card for Insert tab
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 รายละเอียดข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.insert_details
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.BLUE_50
                )
            )
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
    
    def create_delete_election_tab(self):
        """Create tab for deleting data from election_c table - EXACT COPY from original"""
        
        # Search field for finding records to delete
        self.delete_search_field = ft.TextField(
            label="🔍 ค้นหาข้อมูลที่จะลบ",
            hint_text="ใส่เลขบัตรประชาชน ชื่อ หรือนามสกุล",
            width=400,
            on_change=self.search_records_for_delete
        )
        
        # Table to show 5 SELECTED records that can be deleted
        self.delete_candidates_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Select", width=60)),
                ft.DataColumn(ft.Text("Order", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Identity No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Gender", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=60,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Statistics display
        self.delete_stats_text = ft.Text(
            "📊 Database: 0 total records | 0 selected for deletion",
            size=14,
            weight=ft.FontWeight.BOLD
        )
        
        # Delete tab details panel (like ID Show tab)
        self.delete_details = ft.Column([
            ft.Text("👆 Click on any record above to see details...", 
                   size=16, color=ft.Colors.GREY_600)
        ])
        
        return ft.Column([
            # Warning header
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.WARNING, size=30, color=ft.Colors.RED_600),
                            ft.Text("⚠️ ลบข้อมูลการเลือกตั้ง", 
                                   size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_800),
                            ft.Icon(ft.Icons.DELETE_FOREVER, size=30, color=ft.Colors.RED_600),
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        ft.Text("⚠️ คำเตือน: ข้อมูลที่ลบแล้วไม่สามารถกู้คืนได้!", 
                               size=14, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_700)
                    ], spacing=10),
                    padding=15,
                    bgcolor=ft.Colors.RED_100
                )
            ),
            
            # Search and filter card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🔍 ค้นหาข้อมูลที่จะลบ", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        
                        ft.Row([
                            self.delete_search_field
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        
                        ft.Row([
                            ft.ElevatedButton(
                                "☑️ เลือกทั้งหมด",
                                bgcolor=ft.Colors.GREY_600,
                                color=ft.Colors.WHITE,
                                on_click=self.select_all_candidates,
                                icon=ft.Icons.SELECT_ALL
                            ),
                            ft.ElevatedButton(
                                "☐ ยกเลิกเลือกทั้งหมด",
                                bgcolor=ft.Colors.GREY_600,
                                color=ft.Colors.WHITE,
                                on_click=self.unselect_all_candidates,
                                icon=ft.Icons.DESELECT
                            ),
                            ft.ElevatedButton(
                                "🗑️ ลบข้อมูลว่าง",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.delete_empty_records_tab,
                                icon=ft.Icons.CLEAR
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_AROUND)
                    ], spacing=15),
                    padding=20,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Records table card with horizontal and vertical scrolling
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📋 5 ฟิลด์ที่เลือก - ข้อมูลที่พร้อมสำหรับการลบ", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("👆 Scroll horizontally and vertically to see all data", 
                               size=12, color=ft.Colors.RED_600),
                        ft.Divider(),
                        self.delete_stats_text,
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.delete_candidates_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=350,
                            bgcolor=ft.Colors.WHITE,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10
                        )
                    ]),
                    padding=15,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Details card for Delete tab
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 รายละเอียดข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.delete_details
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.ORANGE_50
                )
            ),
            
            # Danger zone - deletion actions
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🚨 โซนอันตราย - การดำเนินการลบ", 
                               size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.RED_800),
                        ft.Divider(color=ft.Colors.RED_400),
                        
                        ft.Row([

                            ft.ElevatedButton(
                                "🗑️ ลบที่เลือก",
                                bgcolor=ft.Colors.RED_600,
                                color=ft.Colors.WHITE,
                                on_click=self.delete_selected_records,
                                icon=ft.Icons.DELETE_FOREVER,
                                style=ft.ButtonStyle(
                                    elevation=4,
                                    shape=ft.RoundedRectangleBorder(radius=8)
                                )
                            ),
                            ft.ElevatedButton(
                                "💀 ลบข้อมูลทั้งหมด",
                                bgcolor=ft.Colors.RED_900,
                                color=ft.Colors.WHITE,
                                on_click=self.delete_all_records_confirm,
                                icon=ft.Icons.DELETE_SWEEP,
                                style=ft.ButtonStyle(
                                    elevation=4,
                                    shape=ft.RoundedRectangleBorder(radius=8)
                                )
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_AROUND),
                        
                        ft.Text("⚠️ การดำเนินการเหล่านี้ไม่สามารถยกเลิกได้! กรุณาระมัดระวัง", 
                               size=12, color=ft.Colors.RED_700, weight=ft.FontWeight.BOLD)
                    ], spacing=10),
                    padding=20,
                    bgcolor=ft.Colors.RED_200
                )
            )
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
    
    def create_update_election_tab(self):
        """Create tab for updating election data in election_c table - EXACT COPY from original"""
        
        # Search field for finding records to update
        self.update_search_field = ft.TextField(
            label="🔍 ค้นหาข้อมูลที่จะแก้ไข",
            hint_text="ใส่เลขบัตรประชาชน ชื่อ หรือนามสกุลเพื่อค้นหาข้อมูล",
            width=400,
            on_change=self.search_records_for_update
        )
        
        # Table to show 5 key fields for records that can be updated
        self.update_candidates_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Order", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Identity No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Action", weight=ft.FontWeight.BOLD))
            ],
            rows=[],
            data_row_max_height=60,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Update form fields (similar to insert but for editing)
        self.update_identity_no = ft.TextField(label="Identity Number", width=300, read_only=True)
        self.update_name_title = ft.Dropdown(
            label="Title",
            options=[
                ft.dropdown.Option("นาย"), ft.dropdown.Option("นาง"), ft.dropdown.Option("น.ส."),
                ft.dropdown.Option("เด็กชาย"), ft.dropdown.Option("เด็กหญิง")
            ],
            width=120
        )
        
        self.update_name = ft.TextField(label="Name", width=200)
        self.update_surname = ft.TextField(label="นามสกุล", width=200)
        self.update_no_address = ft.TextField(label="Address No", width=100)
        self.update_mo_address = ft.TextField(label="หมู่ที่", width=100)
        
        self.update_day_birth = ft.TextField(label="Day", width=80)
        self.update_month = ft.Dropdown(
            label="Month",
            options=[
                ft.dropdown.Option("ม.ค."), ft.dropdown.Option("ก.พ."), ft.dropdown.Option("มี.ค."),
                ft.dropdown.Option("เม.ย."), ft.dropdown.Option("พ.ค."), ft.dropdown.Option("มิ.ย."),
                ft.dropdown.Option("ก.ค."), ft.dropdown.Option("ส.ค."), ft.dropdown.Option("ก.ย."),
                ft.dropdown.Option("ต.ค."), ft.dropdown.Option("พ.ย."), ft.dropdown.Option("ธ.ค.")
            ],
            width=100
        )
        self.update_year = ft.TextField(label="Year (พ.ศ.)", width=100)
        
        self.update_sex = ft.Dropdown(
            label="Gender",
            options=[ft.dropdown.Option("ชาย"), ft.dropdown.Option("หญิง")],
            width=100
        )
        
        self.update_remark = ft.TextField(label="Remark", width=300)
        
        # Hidden field to store the order_no of the record being updated
        self.current_update_order_no = None
        
        # Status display
        self.update_status_text = ft.Text(
            "📝 เลือกข้อมูลจากตารางด้านบนเพื่อแก้ไข",
            size=14,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_600
        )
        
        # Update tab details panel (like ID Show tab)
        self.update_details = ft.Column([
            ft.Text("👆 คลิกบนข้อมูลใดๆ เพื่อดูรายละเอียด...", 
                   size=16, color=ft.Colors.GREY_600)
        ])
        
        return ft.Column([
            # Header card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.EDIT, size=30, color=ft.Colors.BLUE_600),
                        ft.Text("✏️ แก้ไขข้อมูลการเลือกตั้ง", 
                               size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
                        ft.Icon(ft.Icons.SAVE, size=30, color=ft.Colors.BLUE_600),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=15,
                    bgcolor=ft.Colors.BLUE_100
                )
            ),
            
            # Search card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("🔍 ค้นหาข้อมูลที่จะแก้ไข", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        
                        ft.Row([
                            self.update_search_field
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
                    ], spacing=15),
                    padding=20,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Records table card with horizontal and vertical scrolling
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📋 5 ฟิลด์หลัก - ข้อมูลที่มี (คลิกแก้ไขเพื่อเปลี่ยนแปลง)", size=16, weight=ft.FontWeight.BOLD),
                        ft.Text("👆 คลิกบนข้อมูลใดๆ เพื่อดูรายละเอียดเต็ม", 
                               size=12, color=ft.Colors.BLUE_600),
                        ft.Divider(),
                        ft.Container(
                            content=ft.Row([
                                ft.Column([
                                    self.update_candidates_table
                                ], scroll=ft.ScrollMode.ALWAYS, expand=True)
                            ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                            height=250,
                            bgcolor=ft.Colors.WHITE,
                            border=ft.border.all(1, ft.Colors.GREY_300),
                            border_radius=8,
                            padding=10
                        )
                    ]),
                    padding=15,
                    bgcolor=ft.Colors.GREY_100
                )
            ),
            
            # Update form card
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("✏️ แก้ไขข้อมูลที่เลือก", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.update_status_text,
                        
                        # Identity (read-only)
                        ft.Row([self.update_identity_no], alignment=ft.MainAxisAlignment.START),
                        
                        # Basic info
                        ft.Row([
                            self.update_name_title,
                            self.update_name,
                            self.update_surname,
                            self.update_sex
                        ]),
                        
                        # Address
                        ft.Row([
                            self.update_no_address,
                            self.update_mo_address,
                        ]),
                        
                        # Birth date
                        ft.Row([
                            self.update_day_birth,
                            self.update_month,
                            self.update_year
                        ]),
                        
                        # Remark
                        ft.Row([self.update_remark]),
                        
                        # Action buttons
                        ft.Row([
                            ft.ElevatedButton(
                                "💾 บันทึกการเปลี่ยนแปลง",
                                bgcolor=ft.Colors.GREEN_600,
                                color=ft.Colors.WHITE,
                                on_click=self.save_record_update,
                                icon=ft.Icons.SAVE
                            ),
                            ft.ElevatedButton(
                                "🔄 ล้างฟอร์ม",
                                bgcolor=ft.Colors.ORANGE_600,
                                color=ft.Colors.WHITE,
                                on_click=self.clear_update_form,
                                icon=ft.Icons.CLEAR
                            ),
                            ft.ElevatedButton(
                                "❌ ยกเลิกการแก้ไข",
                                bgcolor=ft.Colors.RED_600,
                                color=ft.Colors.WHITE,
                                on_click=self.cancel_record_edit,
                                icon=ft.Icons.CANCEL
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_AROUND)
                    ], spacing=15),
                    padding=20,
                    bgcolor=ft.Colors.GREEN_100
                )
            ),
            
            # Details card for Update tab
            ft.Card(
                elevation=4,
                content=ft.Container(
                    content=ft.Column([
                        ft.Text("📝 รายละเอียดข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                        ft.Divider(),
                        self.update_details
                    ]),
                    padding=20,
                    bgcolor=ft.Colors.PURPLE_50
                )
            )
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
    
    def show_snack_bar(self, message, color=ft.Colors.BLUE_600):
        """Show a snack bar message"""
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message, color=ft.Colors.WHITE),
            bgcolor=color
        )
        self.page.snack_bar.open = True
        self.page.update()
    
    def refresh_all_data(self, e=None):
        """รีเฟรชสถิติ all data"""
        self.show_snack_bar("🔄 รีเฟรชสถิติing all data...", ft.Colors.BLUE_600)
    
    def export_report(self, e=None):
        """Export report"""
        self.show_snack_bar("📄 Export functionality available", ft.Colors.GREEN_600)
    
    def save_election_record(self, e):
        """Save new election record to election_c table"""
        try:
            # Validate required fields
            if not self.insert_identity_no.value or not self.insert_name.value or not self.insert_surname.value:
                self.show_snack_bar("❌ Please fill in Identity Number, Name, and Surname", ft.Colors.RED_600)
                return
            
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Check if identity number already exists
            cursor.execute("SELECT COUNT(id) FROM election_c WHERE identity_no = ?", (self.insert_identity_no.value,))
            if cursor.fetchone()[0] > 0:
                conn.close()
                self.show_snack_bar("❌ Identity number already exists", ft.Colors.RED_600)
                return
            
            # Get next order number
            cursor.execute("SELECT MAX(order_no) FROM election_c")
            max_order = cursor.fetchone()[0] or 0
            new_order = max_order + 1
            
            # Create birthday string
            birthday = ""
            if self.insert_day_birth.value and self.insert_month.value and self.insert_year.value:
                birthday = f"{self.insert_day_birth.value.zfill(2)}/{self.insert_month.value}/{self.insert_year.value}"
            
            # Insert new record
            cursor.execute("""
                INSERT INTO election_c (
                    order_no, identity_no, name_title, name, surname, 
                    no_address, mo_address, day_birth, month, year, sex, remark, birthday
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                new_order,
                self.insert_identity_no.value,
                self.insert_name_title.value or "",
                self.insert_name.value,
                self.insert_surname.value,
                self.insert_no_address.value or "",
                int(self.insert_mo_address.value) if self.insert_mo_address.value else 1,
                int(self.insert_day_birth.value) if self.insert_day_birth.value else None,
                self.insert_month.value or "",
                int(self.insert_year.value) if self.insert_year.value else None,
                self.insert_sex.value or "",
                self.insert_remark.value or "",
                birthday
            ))
            
            conn.commit()
            conn.close()
            
            self.show_snack_bar("✅ Election record saved successfully!", ft.Colors.GREEN_600)
            
            # Add new record to the datagridview immediately
            new_record = {
                'order_no': new_order,
                'identity_no': self.insert_identity_no.value,
                'name': self.insert_name.value,
                'surname': self.insert_surname.value,
                'gender': self.insert_sex.value or ""
            }
            
            # Add to editable data
            self.editable_table_data.insert(0, new_record)
            
            # Add new row to the table
            self.recent_records_table.rows.insert(0, ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(new_record['order_no']))),
                    ft.DataCell(ft.Text(str(new_record['identity_no']))),
                    ft.DataCell(ft.Text(str(new_record['name']))),
                    ft.DataCell(ft.Text(str(new_record['surname']))),
                    ft.DataCell(ft.Text(str(new_record['gender'])))
                ],
                on_select_changed=lambda e, id_num=new_record['identity_no']: self.on_insert_record_select(id_num)
            ))
            
            # Update the table
            self.recent_records_table.update()
            
            # Clear form after successful save
            self.clear_election_form(None)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error saving record: {ex}", ft.Colors.RED_600)
    
    def clear_election_form(self, e):
        """Clear all form fields"""
        self.insert_identity_no.value = ""
        self.insert_name_title.value = None
        self.insert_name.value = ""
        self.insert_surname.value = ""
        self.insert_no_address.value = ""
        self.insert_mo_address.value = "1"
        self.insert_day_birth.value = ""
        self.insert_month.value = None
        self.insert_year.value = ""
        self.insert_sex.value = None
        self.insert_remark.value = ""
        
        # Update all fields
        self.insert_identity_no.update()
        self.insert_name_title.update()
        self.insert_name.update()
        self.insert_surname.update()
        self.insert_no_address.update()
        self.insert_mo_address.update()
        self.insert_day_birth.update()
        self.insert_month.update()
        self.insert_year.update()
        self.insert_sex.update()
        self.insert_remark.update()
        
        self.show_snack_bar("🔄 Form cleared", ft.Colors.BLUE_600)
    
    def load_recent_records(self, e):
        """Load recently added records from election_c table"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get last 10 records with ALL fields ordered by order_no descending
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                ORDER BY order_no DESC 
                LIMIT 10
            """)
            
            recent_records = cursor.fetchall()
            conn.close()
            
            # Clear existing rows and data
            self.recent_records_table.rows.clear()
            self.editable_table_data.clear()
            
            # Add new rows with 5 key fields only
            for record in recent_records:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, sex, remark, birthday = record
                
                # Store editable data
                editable_record = {
                    'order_no': order_no,
                    'identity_no': identity_no,
                    'name': name,
                    'surname': surname,
                    'gender': sex
                }
                self.editable_table_data.append(editable_record)
                
                self.recent_records_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(sex or "")))
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_insert_record_select(id_num)
                    )
                )
            
            self.recent_records_table.update()
            self.show_snack_bar(f"📋 Loaded {len(recent_records)} recent records", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading recent records: {ex}", ft.Colors.RED_600)

    # DELETE TAB FUNCTIONS
    def search_records_for_delete(self, e):
        """Search records for deletion based on search term"""
        search_term = self.delete_search_field.value.strip()
        if not search_term:
            return
        
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Search records with ALL fields
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no LIKE ? OR name LIKE ? OR surname LIKE ?
                ORDER BY order_no
                LIMIT 50
            """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
            
            records = cursor.fetchall()
            conn.close()
            
            # Update table with search results
            self.delete_candidate_records = []
            self.delete_selected_records_set = set()
            self.delete_candidates_table.rows.clear()
            
            for record in records:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, sex, remark, birthday = record
                
                record_info = {
                    'order_no': order_no,
                    'identity_no': identity_no,
                    'title': title,
                    'name': name,
                    'surname': surname,
                    'address_no': address_no,
                    'mo_address': mo_address,
                    'day_birth': day_birth,
                    'month': month,
                    'year': year,
                    'sex': sex,
                    'remark': remark,
                    'birthday': birthday
                }
                self.delete_candidate_records.append(record_info)
                
                checkbox = ft.Checkbox(
                    value=False,
                    on_change=lambda e, idx=len(self.delete_candidate_records)-1: self.toggle_record_selection(e, idx)
                )
                
                self.delete_candidates_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(checkbox),
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(sex or "")))
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_delete_record_select(id_num)
                    )
                )
            
            self.delete_candidates_table.update()
            self.update_delete_stats()
            
            self.show_snack_bar(f"🔍 Found {len(records)} matching records", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Search error: {ex}", ft.Colors.RED_600)
    
    def load_all_delete_candidates(self, e):
        """Load all records from election_c table for deletion consideration"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get all records with ALL fields
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                ORDER BY order_no
                LIMIT 100
            """)
            
            records = cursor.fetchall()
            
            # Get total count
            cursor.execute("SELECT COUNT(id) FROM election_c")
            total_count = cursor.fetchone()[0]
            
            conn.close()
            
            # Store records for selection tracking
            self.delete_candidate_records = []
            self.delete_selected_records_set = set()
            
            # Clear and populate table
            self.delete_candidates_table.rows.clear()
            
            for record in records:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, sex, remark, birthday = record
                
                # Store record info with ALL fields
                record_info = {
                    'order_no': order_no,
                    'identity_no': identity_no,
                    'title': title,
                    'name': name,
                    'surname': surname,
                    'address_no': address_no,
                    'mo_address': mo_address,
                    'day_birth': day_birth,
                    'month': month,
                    'year': year,
                    'sex': sex,
                    'remark': remark,
                    'birthday': birthday
                }
                self.delete_candidate_records.append(record_info)
                
                # Create checkbox for selection
                checkbox = ft.Checkbox(
                    value=False,
                    on_change=lambda e, idx=len(self.delete_candidate_records)-1: self.toggle_record_selection(e, idx)
                )
                
                self.delete_candidates_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(checkbox),
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(sex or "")))
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_delete_record_select(id_num)
                    )
                )
            
            self.delete_candidates_table.update()
            self.update_delete_stats()
            
            self.show_snack_bar(f"📋 Loaded {len(records)} records for deletion review", ft.Colors.BLUE_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading records: {ex}", ft.Colors.RED_600)
    
    def toggle_record_selection(self, e, record_index):
        """Toggle selection of a record for deletion"""
        if e.control.value:
            self.delete_selected_records_set.add(record_index)
        else:
            self.delete_selected_records_set.discard(record_index)
        
        self.update_delete_stats()
    
    def update_delete_stats(self):
        """Update the statistics display"""
        total_in_db = len(getattr(self, 'delete_candidate_records', []))
        selected_count = len(getattr(self, 'delete_selected_records_set', set()))
        
        self.delete_stats_text.value = f"📊 Showing: {total_in_db} records | Selected for deletion: {selected_count}"
        self.delete_stats_text.update()
    
    def select_all_candidates(self, e):
        """Select all loaded records for deletion"""
        self.delete_selected_records_set = set(range(len(self.delete_candidate_records)))
        
        # Update all checkboxes
        for i, row in enumerate(self.delete_candidates_table.rows):
            checkbox = row.cells[0].content
            checkbox.value = True
            checkbox.update()
        
        self.update_delete_stats()
        self.show_snack_bar("☑️ All visible records selected", ft.Colors.BLUE_600)
    
    def unselect_all_candidates(self, e):
        """Unselect all records"""
        self.delete_selected_records_set.clear()
        
        # Update all checkboxes
        for row in self.delete_candidates_table.rows:
            checkbox = row.cells[0].content
            checkbox.value = False
            checkbox.update()
        
        self.update_delete_stats()
        self.show_snack_bar("☐ All records unselected", ft.Colors.BLUE_600)
    
    def delete_empty_records_tab(self, e):
        """Delete empty records from delete tab"""
        self.show_snack_bar("🗑️ Delete empty records function connected to election_c", ft.Colors.ORANGE_600)
    
    def delete_selected_records(self, e):
        """Delete selected records with warning and confirmation"""
        try:
            # Check for selected records using the selection tracking
            if not hasattr(self, 'delete_selected_records_set') or not self.delete_selected_records_set:
                self.show_snack_bar("⚠️ No records selected for deletion. Please select records first.", ft.Colors.ORANGE_600)
                return
            
            if not hasattr(self, 'delete_candidate_records') or not self.delete_candidate_records:
                self.show_snack_bar("⚠️ No records loaded. Please load records first.", ft.Colors.ORANGE_600)
                return
            
            # Get selected record details
            selected_records = []
            for index in self.delete_selected_records_set:
                if index < len(self.delete_candidate_records):
                    record = self.delete_candidate_records[index]
                    selected_records.append({
                        'identity_no': record['identity_no'],
                        'name': record['name'],
                        'surname': record['surname'],
                        'order_no': record['order_no']
                    })
            
            # Execute deletion directly without confirmation
            count = len(selected_records)
            self.show_snack_bar(f"🗑️ Deleting {count} selected records...", ft.Colors.RED_600)
            
            # Execute deletion immediately
            self.execute_selected_records_deletion(selected_records)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error preparing deletion: {ex}", ft.Colors.RED_600)
    
    def cancel_delete_confirmation(self):
        """Cancel the delete confirmation"""
        self.delete_details.controls.clear()
        self.delete_details.controls.append(
            ft.Text("👆 Click on any record above to see details...", 
                   size=16, color=ft.Colors.GREY_600)
        )
        self.delete_details.update()
        self.show_snack_bar("❌ Deletion cancelled", ft.Colors.BLUE_600)
    
    def confirm_delete_selected(self, selected_records):
        """Confirm and execute deletion of selected records"""
        try:
            # Check confirmation text
            if not hasattr(self, 'delete_confirmation_text_field') or \
               self.delete_confirmation_text_field.value != "DELETE SELECTED":
                self.show_snack_bar("❌ Please type 'DELETE SELECTED' exactly to confirm", ft.Colors.RED_600)
                return
            
            # Execute deletion
            self.execute_selected_records_deletion(selected_records)
            
            # Clear confirmation
            self.cancel_delete_confirmation()
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error confirming deletion: {ex}", ft.Colors.RED_600)
    
    def execute_selected_records_deletion(self, selected_records):
        """Execute deletion of selected records"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            deleted_count = 0
            failed_deletions = []
            
            for record in selected_records:
                try:
                    cursor.execute("DELETE FROM election_c WHERE identity_no = ?", (record['identity_no'],))
                    if cursor.rowcount > 0:
                        deleted_count += 1
                    else:
                        failed_deletions.append(record['identity_no'])
                except Exception as e:
                    failed_deletions.append(f"{record['identity_no']} (Error: {str(e)})")
            
            conn.commit()
            conn.close()
            
            # รีเฟรชสถิติ the delete table
            self.load_all_delete_candidates(None)
            
            # Update statistics
            try:
                conn = sqlite3.connect('demo_voters.db')
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(id) FROM election_c")
                total_records = cursor.fetchone()[0]
                conn.close()
                
                self.delete_stats_text.value = f"📊 Database: {total_records:,} total records | {deleted_count} records deleted"
                self.delete_stats_text.update()
            except:
                pass
            
            # Show result
            if failed_deletions:
                self.show_snack_bar(f"⚠️ Deleted {deleted_count} records, {len(failed_deletions)} failed", ft.Colors.ORANGE_600)
            else:
                self.show_snack_bar(f"✅ Successfully deleted {deleted_count} selected records", ft.Colors.GREEN_600)
            
            # รีเฟรชสถิติ other tabs
            try:
                self.load_recent_records()  # รีเฟรชสถิติ Insert tab
            except:
                pass
                
        except Exception as ex:
            self.show_snack_bar(f"❌ Error during deletion: {ex}", ft.Colors.RED_600)
    

    def delete_all_records_confirm(self, e):
        """Delete ALL records directly without confirmation"""
        try:
            # First get the total count
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(id) FROM election_c")
            total_records = cursor.fetchone()[0]
            conn.close()
            
            if total_records == 0:
                self.show_snack_bar("ℹ️ No records to delete - table is already empty", ft.Colors.BLUE_600)
                return
            
            # Execute deletion directly
            self.show_snack_bar(f"🗑️ Deleting ALL {total_records:,} records...", ft.Colors.RED_800)
            self.execute_delete_all_records(total_records)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error checking records: {ex}", ft.Colors.RED_600)
    
    def cancel_delete_all_confirmation(self):
        """Cancel the delete all confirmation"""
        self.delete_details.controls.clear()
        self.delete_details.controls.append(
            ft.Text("👆 Click on any record above to see details...", 
                   size=16, color=ft.Colors.GREY_600)
        )
        self.delete_details.update()
        self.show_snack_bar("✅ Delete All operation cancelled - Data is safe!", ft.Colors.GREEN_600)
    
    def confirm_delete_all(self, expected_count):
        """Confirm and execute deletion of all records"""
        try:
            # Check confirmation text
            if not hasattr(self, 'delete_all_confirmation_text_field') or \
               self.delete_all_confirmation_text_field.value != "DELETE ALL DATA":
                self.show_snack_bar("❌ Please type 'DELETE ALL DATA' exactly to confirm", ft.Colors.RED_600)
                return
            
            # Execute deletion
            self.execute_delete_all_records(expected_count)
            
            # Clear confirmation
            self.cancel_delete_all_confirmation()
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error confirming delete all: {ex}", ft.Colors.RED_600)
    

    
    def execute_delete_all_records(self, expected_count):
        """Execute the actual deletion after all confirmations"""
        try:
            # Show progress
            self.show_snack_bar("🔄 Deleting all records... Please wait...", ft.Colors.ORANGE_600)
            
            # Execute deletion
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get final count before deletion
            cursor.execute("SELECT COUNT(id) FROM election_c")
            actual_count = cursor.fetchone()[0]
            
            if actual_count != expected_count:
                conn.close()
                self.show_snack_bar(f"⚠️ Record count changed! Expected {expected_count}, found {actual_count}. Operation cancelled.", ft.Colors.ORANGE_600)
                return
            
            # Perform the deletion
            cursor.execute("DELETE FROM election_c")
            deleted_count = cursor.rowcount
            
            # Reset auto-increment counter
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='election_c'")
            
            conn.commit()
            conn.close()
            
            # Update displays
            self.delete_candidates_table.rows.clear()
            self.delete_candidates_table.update()
            
            self.delete_stats_text.value = "📊 Database: 0 total records | All data deleted"
            self.delete_stats_text.update()
            
            # Clear confirmation field
            if hasattr(self, 'delete_confirmation_field'):
                self.delete_confirmation_field.value = ""
                self.delete_confirmation_field.update()
            
            # Success message
            self.show_snack_bar(f"💀 ALL DATA DELETED! {deleted_count:,} records permanently removed from election_c table", ft.Colors.RED_800)
            
            # รีเฟรชสถิติ other tabs if they have data
            try:
                self.load_recent_records()  # รีเฟรชสถิติ Insert tab
            except:
                pass
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error during deletion: {ex}", ft.Colors.RED_600)
    
    # UPDATE TAB FUNCTIONS
    def search_records_for_update(self, e):
        """Search records for updating based on search term"""
        search_term = self.update_search_field.value.strip()
        if not search_term:
            return
        
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Search records with ALL fields
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no LIKE ? OR name LIKE ? OR surname LIKE ?
                ORDER BY order_no
                LIMIT 50
            """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
            
            records = cursor.fetchall()
            conn.close()
            
            # Clear and populate table
            self.update_candidates_table.rows.clear()
            
            for record in records:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, sex, remark, birthday = record
                
                # Create edit button for each record
                edit_button = ft.ElevatedButton(
                    "✏️ Edit",
                    bgcolor=ft.Colors.BLUE_600,
                    color=ft.Colors.WHITE,
                    width=80,
                    height=35,
                    on_click=lambda e, order=order_no: self.load_record_for_edit(e, order)
                )
                
                self.update_candidates_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(edit_button)
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_update_record_select(id_num)
                    )
                )
            
            self.update_candidates_table.update()
            
            self.show_snack_bar(f"🔍 Found {len(records)} matching records", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Search error: {ex}", ft.Colors.RED_600)
    
    def load_all_update_candidates(self, e):
        """Load all records from election_c table for update consideration"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get all records with ALL fields
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday
                FROM election_c 
                ORDER BY order_no
                LIMIT 100
            """)
            
            records = cursor.fetchall()
            conn.close()
            
            # Clear and populate table
            self.update_candidates_table.rows.clear()
            
            for record in records:
                order_no, identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, sex, remark, birthday = record
                
                # Create edit button for each record
                edit_button = ft.ElevatedButton(
                    "✏️ Edit",
                    bgcolor=ft.Colors.BLUE_600,
                    color=ft.Colors.WHITE,
                    width=80,
                    height=35,
                    on_click=lambda e, order=order_no: self.load_record_for_edit(e, order)
                )
                
                self.update_candidates_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(order_no or ""))),
                            ft.DataCell(ft.Text(str(identity_no or "N/A"))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(edit_button)
                        ],
                        on_select_changed=lambda e, id_num=identity_no: self.on_update_record_select(id_num)
                    )
                )
            
            self.update_candidates_table.update()
            
            self.show_snack_bar(f"📋 Loaded {len(records)} records for update", ft.Colors.BLUE_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading records: {ex}", ft.Colors.RED_600)
    
    def load_record_for_edit(self, e, order_no):
        """Show edit modal dialog instead of using form panel"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get full record details
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, 
                       no_address, mo_address, day_birth, month, year, sex, remark
                FROM election_c 
                WHERE order_no = ?
            """, (order_no,))
            
            record = cursor.fetchone()
            conn.close()
            
            if record:
                # Show edit modal instead of filling form panel
                self.show_edit_modal_dialog(record)
            else:
                self.show_snack_bar("❌ Record not found", ft.Colors.RED_600)
                
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading record: {ex}", ft.Colors.RED_600)
    
    def show_desktop_modal_dialog(self, title, content, width=500, height=600):
        """Show modal overlay dialog for desktop version - similar to mobile with drag functionality"""
        print(f"🔧 DEBUG: Creating desktop modal dialog: {title}")
        
        # Drag state variables
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.modal_start_left = (1400-width)//2
        self.modal_start_top = 50
        
        def close_modal(e):
            print("🔧 DEBUG: Closing desktop modal overlay")
            if hasattr(self, 'desktop_modal_overlay') and self.desktop_modal_overlay in self.page.overlay:
                self.page.overlay.remove(self.desktop_modal_overlay)
                self.page.update()
        
        def on_pan_start(e):
            """Start dragging the modal"""
            self.is_dragging = True
            self.drag_start_x = e.global_x
            self.drag_start_y = e.global_y
            print(f"🔧 DEBUG: Drag started at ({e.global_x}, {e.global_y})")
        
        def on_pan_update(e):
            """Update modal position during drag"""
            if self.is_dragging:
                # Calculate new position
                delta_x = e.global_x - self.drag_start_x
                delta_y = e.global_y - self.drag_start_y
                
                new_left = max(0, min(1400-width, self.modal_start_left + delta_x))
                new_top = max(0, min(800-height, self.modal_start_top + delta_y))
                
                # Update modal position
                self.desktop_modal_overlay.left = new_left
                self.desktop_modal_overlay.top = new_top
                self.desktop_modal_overlay.update()
        
        def on_pan_end(e):
            """End dragging and save new position"""
            if self.is_dragging:
                self.is_dragging = False
                self.modal_start_left = self.desktop_modal_overlay.left
                self.modal_start_top = self.desktop_modal_overlay.top
                print(f"🔧 DEBUG: Drag ended at ({self.modal_start_left}, {self.modal_start_top})")
        
        # Split content into lines for ListView
        content_lines = content.split('\n')
        
        # Create ListView items
        list_items = []
        for line in content_lines:
            if line.strip():  # Skip empty lines
                list_items.append(
                    ft.ListTile(
                        title=ft.Text(line, size=12, selectable=True),
                        dense=True
                    )
                )
            else:
                list_items.append(ft.Divider(height=5))
        
        # Create modal overlay with ListView
        self.desktop_modal_overlay = ft.Container(
            content=ft.Column([
                # Modal header with drag functionality
                ft.GestureDetector(
                    content=ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.DRAG_INDICATOR, color=ft.Colors.WHITE, size=16),
                            ft.Text(title, size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, expand=True),
                            ft.IconButton(
                                ft.Icons.CLOSE,
                                icon_color=ft.Colors.WHITE,
                                on_click=close_modal
                            )
                        ]),
                        bgcolor=ft.Colors.BLUE_800,
                        padding=ft.padding.all(15),
                        border_radius=ft.border_radius.only(top_left=10, top_right=10)
                    ),
                    on_pan_start=on_pan_start,
                    on_pan_update=on_pan_update,
                    on_pan_end=on_pan_end
                ),
                # Modal content with ListView
                ft.Container(
                    content=ft.ListView(
                        controls=list_items,
                        expand=True,
                        spacing=2
                    ),
                    height=height-80,
                    padding=ft.padding.all(10),
                    bgcolor=ft.Colors.WHITE
                ),
                # Modal footer
                ft.Container(
                    content=ft.Row([
                        ft.ElevatedButton(
                            "ปิด",
                            icon=ft.Icons.CLOSE,
                            on_click=close_modal,
                            bgcolor=ft.Colors.BLUE_600,
                            color=ft.Colors.WHITE
                        )
                    ], alignment=ft.MainAxisAlignment.END),
                    bgcolor=ft.Colors.GREY_100,
                    padding=ft.padding.all(10),
                    border_radius=ft.border_radius.only(bottom_left=10, bottom_right=10)
                )
            ]),
            width=width,
            height=height,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.BLACK26
            ),
            margin=ft.margin.all(20),
            top=50,
            left=(1400-width)//2,  # Center horizontally for desktop
            right=(1400-width)//2
        )
        
        # Add to page overlay
        self.page.overlay.append(self.desktop_modal_overlay)
        self.page.update()
        print(f"🔧 DEBUG: Desktop modal overlay added to page")
    
    def show_edit_modal_dialog(self, record):
        """Show edit form in modal dialog for desktop version"""
        order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark = record
        print(f"🔧 DEBUG: Creating desktop edit modal for: {identity_no}")
        
        # Drag state variables for edit modal
        self.edit_modal_is_dragging = False
        self.edit_modal_drag_start_x = 0
        self.edit_modal_drag_start_y = 0
        self.edit_modal_start_left = (1400-500)//2
        self.edit_modal_start_top = 30
        
        def close_modal(e):
            print("🔧 DEBUG: Closing desktop edit modal")
            if hasattr(self, 'edit_modal_overlay') and self.edit_modal_overlay in self.page.overlay:
                self.page.overlay.remove(self.edit_modal_overlay)
                self.page.update()
        
        def on_edit_pan_start(e):
            """Start dragging the edit modal"""
            self.edit_modal_is_dragging = True
            self.edit_modal_drag_start_x = e.global_x
            self.edit_modal_drag_start_y = e.global_y
            print(f"🔧 DEBUG: Edit modal drag started at ({e.global_x}, {e.global_y})")
        
        def on_edit_pan_update(e):
            """Update edit modal position during drag"""
            if self.edit_modal_is_dragging:
                # Calculate new position
                delta_x = e.global_x - self.edit_modal_drag_start_x
                delta_y = e.global_y - self.edit_modal_drag_start_y
                
                new_left = max(0, min(1400-500, self.edit_modal_start_left + delta_x))
                new_top = max(0, min(800-700, self.edit_modal_start_top + delta_y))
                
                # Update modal position
                self.edit_modal_overlay.left = new_left
                self.edit_modal_overlay.top = new_top
                self.edit_modal_overlay.update()
        
        def on_edit_pan_end(e):
            """End dragging and save new position"""
            if self.edit_modal_is_dragging:
                self.edit_modal_is_dragging = False
                self.edit_modal_start_left = self.edit_modal_overlay.left
                self.edit_modal_start_top = self.edit_modal_overlay.top
                print(f"🔧 DEBUG: Edit modal drag ended at ({self.edit_modal_start_left}, {self.edit_modal_start_top})")
        
        def save_update(e):
            try:
                # Get values from form fields
                name_title_value = self.modal_name_title.value
                name_value = self.modal_name.value
                surname_value = self.modal_surname.value
                sex_value = self.modal_sex.value
                address_value = self.modal_no_address.value
                mo_address_value = self.modal_mo_address.value
                day_birth_value = self.modal_day_birth.value
                month_value = self.modal_month.value
                year_value = self.modal_year.value
                remark_value = self.modal_remark.value
                
                # Update database
                conn = sqlite3.connect('demo_voters.db')
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE election_c 
                    SET name_title = ?, name = ?, surname = ?, sex = ?, 
                        no_address = ?, mo_address = ?, day_birth = ?, 
                        month = ?, year = ?, remark = ?
                    WHERE order_no = ?
                """, (name_title_value, name_value, surname_value, sex_value, 
                      address_value, mo_address_value, day_birth_value, 
                      month_value, year_value, remark_value, order_no))
                
                conn.commit()
                conn.close()
                
                self.show_snack_bar("✅ แก้ไขข้อมูลสำเร็จ", ft.Colors.GREEN_600)
                close_modal(e)
                # รีเฟรชสถิติ the current tab
                self.load_all_update_candidates(e)
                
            except Exception as ex:
                self.show_snack_bar(f"❌ Error saving: {ex}", ft.Colors.RED_600)
        
        # Create form fields with current values
        self.modal_name_title = ft.Dropdown(
            label="คำนำหน้า",
            options=[
                ft.dropdown.Option("นาย"), ft.dropdown.Option("นาง"), ft.dropdown.Option("น.ส."),
                ft.dropdown.Option("เด็กชาย"), ft.dropdown.Option("เด็กหญิง")
            ],
            value=name_title or "",
            width=120
        )
        
        self.modal_name = ft.TextField(
            label="ชื่อ", 
            value=name or "", 
            width=200,
            border_color=ft.Colors.BLUE_400
        )
        
        self.modal_surname = ft.TextField(
            label="นามสกุล", 
            value=surname or "", 
            width=200,
            border_color=ft.Colors.BLUE_400
        )
        
        self.modal_sex = ft.Dropdown(
            label="เพศ",
            options=[
                ft.dropdown.Option("ชาย"), 
                ft.dropdown.Option("หญิง")
            ],
            value=sex or "",
            width=100
        )
        
        self.modal_no_address = ft.TextField(
            label="บ้านเลขที่", 
            value=no_address or "", 
            width=100,
            border_color=ft.Colors.BLUE_400
        )
        
        self.modal_mo_address = ft.TextField(
            label="หมู่ที่", 
            value=mo_address or "", 
            width=100,
            border_color=ft.Colors.BLUE_400
        )
        
        self.modal_day_birth = ft.TextField(
            label="วัน", 
            value=day_birth or "", 
            width=80,
            border_color=ft.Colors.BLUE_400
        )
        
        self.modal_month = ft.Dropdown(
            label="เดือน",
            options=[
                ft.dropdown.Option("ม.ค."), ft.dropdown.Option("ก.พ."), ft.dropdown.Option("มี.ค."),
                ft.dropdown.Option("เม.ย."), ft.dropdown.Option("พ.ค."), ft.dropdown.Option("มิ.ย."),
                ft.dropdown.Option("ก.ค."), ft.dropdown.Option("ส.ค."), ft.dropdown.Option("ก.ย."),
                ft.dropdown.Option("ต.ค."), ft.dropdown.Option("พ.ย."), ft.dropdown.Option("ธ.ค.")
            ],
            value=month or "",
            width=100
        )
        
        self.modal_year = ft.TextField(
            label="ปี (พ.ศ.)", 
            value=year or "", 
            width=100,
            border_color=ft.Colors.BLUE_400
        )
        
        self.modal_remark = ft.TextField(
            label="หมายเหตุ", 
            value=remark or "", 
            width=400,
            multiline=True,
            min_lines=2,
            max_lines=3,
            border_color=ft.Colors.BLUE_400
        )
        
        # Create form in ListView for better scrolling
        form_items = [
            ft.ListTile(
                title=ft.Text("📝 แก้ไขข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                subtitle=ft.Text(f"รหัสประชาชน: {identity_no}", size=12, color=ft.Colors.BLUE_600)
            ),
            ft.Divider(),
            ft.ListTile(
                title=ft.Row([
                    self.modal_name_title,
                    self.modal_name,
                    self.modal_surname,
                    self.modal_sex
                ])
            ),
            ft.ListTile(
                title=ft.Row([
                    self.modal_no_address,
                    self.modal_mo_address
                ])
            ),
            ft.ListTile(
                title=ft.Row([
                    self.modal_day_birth,
                    self.modal_month,
                    self.modal_year
                ])
            ),
            ft.ListTile(title=self.modal_remark),
            ft.Divider(),
            ft.ListTile(
                title=ft.Row([
                    ft.ElevatedButton(
                        "💾 บันทึก",
                        icon=ft.Icons.SAVE,
                        on_click=save_update,
                        bgcolor=ft.Colors.GREEN_600,
                        color=ft.Colors.WHITE,
                        expand=True
                    ),
                    ft.Container(width=10),
                    ft.ElevatedButton(
                        "❌ ยกเลิก",
                        icon=ft.Icons.CANCEL,
                        on_click=close_modal,
                        bgcolor=ft.Colors.RED_600,
                        color=ft.Colors.WHITE,
                        expand=True
                    )
                ])
            )
        ]
        
        # Create modal overlay
        self.edit_modal_overlay = ft.Container(
            content=ft.Column([
                # Modal header with drag functionality
                ft.GestureDetector(
                    content=ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.DRAG_INDICATOR, color=ft.Colors.WHITE, size=16),
                            ft.Text("✏️ แก้ไขข้อมูล", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, expand=True),
                            ft.IconButton(
                                ft.Icons.CLOSE,
                                icon_color=ft.Colors.WHITE,
                                on_click=close_modal
                            )
                        ]),
                        bgcolor=ft.Colors.PURPLE_800,
                        padding=ft.padding.all(15),
                        border_radius=ft.border_radius.only(top_left=10, top_right=10)
                    ),
                    on_pan_start=on_edit_pan_start,
                    on_pan_update=on_edit_pan_update,
                    on_pan_end=on_edit_pan_end
                ),
                # Modal content with form
                ft.Container(
                    content=ft.ListView(
                        controls=form_items,
                        expand=True,
                        spacing=5
                    ),
                    height=520,
                    padding=ft.padding.all(10),
                    bgcolor=ft.Colors.WHITE
                )
            ]),
            width=600,
            height=600,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.BLACK26
            ),
            margin=ft.margin.all(20),
            top=50,
            left=(1400-600)//2,  # Center horizontally for desktop
            right=(1400-600)//2
        )
        
        # Add to page overlay
        self.page.overlay.append(self.edit_modal_overlay)
        self.page.update()
        print(f"🔧 DEBUG: Desktop edit modal overlay added to page")
    
    def save_record_update(self, e):
        """Save record updates to database"""
        self.show_snack_bar("💾 Save update function connected to election_c", ft.Colors.GREEN_600)
    
    def clear_update_form(self, e):
        """Clear the update form"""
        self.show_snack_bar("🔄 Clear update form function connected", ft.Colors.ORANGE_600)
    
    def cancel_record_edit(self, e):
        """Cancel record editing"""
        self.show_snack_bar("❌ Cancel edit function connected", ft.Colors.RED_600)

    # DETAIL PANEL FUNCTIONS FOR EACH TAB
    def on_insert_record_select(self, identity_no):
        """Handle record selection in Insert tab (show details like ID Show tab)"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get full details for selected identity
            cursor.execute("""
                SELECT * FROM election_c WHERE identity_no = ?
            """, (identity_no,))
            identity = cursor.fetchone()
            
            if identity:
                # Get column names
                cursor.execute("PRAGMA table_info(election_c)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Clear details
                self.insert_details.controls.clear()
                
                # Add title
                self.insert_details.controls.append(
                    ft.Text(f"📝 แท็บเพิ่มข้อมูล - รายละเอียดข้อมูล: {identity_no}", 
                           weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.GREEN_700)
                )
                self.insert_details.controls.append(ft.Divider())
                
                # Add all field details
                for i, value in enumerate(identity):
                    if i < len(columns):
                        field_name = columns[i]
                        display_value = str(value) if value is not None else "N/A"
                        
                        self.insert_details.controls.append(
                            ft.Row([
                                ft.Text(f"{field_name}:", weight=ft.FontWeight.BOLD, width=120),
                                ft.Text(display_value, selectable=True)
                            ])
                        )
            else:
                self.insert_details.controls = [
                    ft.Text(f"❌ ไม่พบรายละเอียดสำหรับเลขประจำตัว: {identity_no}", 
                           color=ft.Colors.RED_600)
                ]
            
            conn.close()
            self.page.update()
            
        except Exception as e:
            print(f"Error loading insert details: {e}")
            self.insert_details.controls = [
                ft.Text(f"❌ เกิดข้อผิดพลาดในการโหลดรายละเอียด: {e}", color=ft.Colors.RED_600)
            ]
            self.page.update()
    
    def on_delete_record_select(self, identity_no):
        """Handle record selection in Delete tab (show details like ID Show tab)"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get full details for selected identity
            cursor.execute("""
                SELECT * FROM election_c WHERE identity_no = ?
            """, (identity_no,))
            identity = cursor.fetchone()
            
            if identity:
                # Get column names
                cursor.execute("PRAGMA table_info(election_c)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Clear details
                self.delete_details.controls.clear()
                
                # Add title
                self.delete_details.controls.append(
                    ft.Text(f"🗑️ แท็บลบข้อมูล - รายละเอียดข้อมูล: {identity_no}", 
                           weight=ft.FontWeight.BOLD, size=16, color=ft.Colors.RED_700)
                )
                self.delete_details.controls.append(ft.Divider())
                
                # Add all field details
                for i, value in enumerate(identity):
                    if i < len(columns):
                        field_name = columns[i]
                        display_value = str(value) if value is not None else "N/A"
                        
                        self.delete_details.controls.append(
                            ft.Row([
                                ft.Text(f"{field_name}:", weight=ft.FontWeight.BOLD, width=120),
                                ft.Text(display_value, selectable=True)
                            ])
                        )
            else:
                self.delete_details.controls = [
                    ft.Text(f"❌ ไม่พบรายละเอียดสำหรับเลขประจำตัว: {identity_no}", 
                           color=ft.Colors.RED_600)
                ]
            
            conn.close()
            self.page.update()
            
        except Exception as e:
            print(f"Error loading delete details: {e}")
            self.delete_details.controls = [
                ft.Text(f"❌ เกิดข้อผิดพลาดในการโหลดรายละเอียด: {e}", color=ft.Colors.RED_600)
            ]
            self.page.update()
    
    def on_update_record_select(self, identity_no):
        """Show details modal dialog instead of updating details panel"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get full details for selected identity
            cursor.execute("""
                SELECT * FROM election_c WHERE identity_no = ?
            """, (identity_no,))
            identity = cursor.fetchone()
            
            if identity:
                # Get column names
                cursor.execute("PRAGMA table_info(election_c)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Format details for modal dialog
                details_text = f"✏️ แท็บแก้ไขข้อมูล - รายละเอียดข้อมูล: {identity_no}\n\n"
                details_text += "=" * 50 + "\n\n"
                
                # Add all field details
                for i, value in enumerate(identity):
                    if i < len(columns):
                        field_name = columns[i]
                        display_value = str(value) if value is not None else "N/A"
                        details_text += f"{field_name}: {display_value}\n"
                
                # Show details in modal dialog
                self.show_desktop_modal_dialog(f"🆔 รายละเอียดข้อมูล: {identity_no}", details_text, 500, 600)
            else:
                self.show_snack_bar(f"❌ ไม่พบรายละเอียดสำหรับเลขประจำตัว: {identity_no}", ft.Colors.RED_600)
            
            conn.close()
            
        except Exception as e:
            print(f"Error loading update details: {e}")
            self.show_snack_bar(f"❌ เกิดข้อผิดพลาดในการโหลดรายละเอียด: {e}", ft.Colors.RED_600)

    # DATABASE FUNCTIONS FOR TABS 1-5 (Import, Family, House, Birth, Activity)
    def on_file_picked(self, e):
        """Handle file picker result"""
        if e.files:
            self.selected_file = e.files[0].path
            self.file_path_text.value = f"📁 {e.files[0].name}"
            self.file_path_text.update()
        else:
            self.selected_file = None
            self.file_path_text.value = "ยังไม่ได้เลือกไฟล์"
            self.file_path_text.update()

    def import_data(self, e):
        """Import data from Excel file"""
        if not hasattr(self, 'selected_file') or not self.selected_file:
            self.show_snack_bar("Please select an Excel file first", ft.Colors.ORANGE_600)
            return
        
        if not Path(self.selected_file).exists():
            self.show_snack_bar("Selected file does not exist", ft.Colors.RED_600)
            return
        
        # Show progress
        self.progress_bar.visible = True
        self.progress_bar.update()
        
        try:
            # Import using database module
            result = self.db.import_from_excel(self.selected_file)
            
            if result['success']:
                message = f"✅ Import completed successfully!\nImported: {result['imported_count']} records"
                self.show_snack_bar(message, ft.Colors.GREEN_600)
            else:
                message = f"❌ Import failed: {result['error']}"
                self.show_snack_bar(message, ft.Colors.RED_600)
                
            self.import_status.value = message
            self.import_status.update()
            
        except Exception as ex:
            error_msg = f"❌ Import error: {ex}"
            self.import_status.value = error_msg
            self.import_status.update()
            self.show_snack_bar(error_msg, ft.Colors.RED_600)
        
        finally:
            self.progress_bar.visible = False
            self.progress_bar.update()
            self.update_stats(None)

    def update_stats(self, e):
        """Update database statistics"""
        try:
            # Only update stats if stats_text exists (Import tab was removed)
            if not hasattr(self, 'stats_text'):
                return
                
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM voters")
            total_voters = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT surname) FROM voters")
            total_families = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT house_id) FROM voters")
            total_houses = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT birthplace_code) FROM voters")
            total_birthplaces = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(age) FROM voters WHERE age IS NOT NULL")
            avg_age = cursor.fetchone()[0] or 0
            
            stats_text = f"""📊 Database Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 Total Voters: {total_voters:,}
👨‍👩‍👧‍👦 Total Families: {total_families:,}
🏠 Total Houses: {total_houses:,}
🌍 Total Birthplaces: {total_birthplaces:,}
🎂 Average Age: {avg_age:.1f} years

💾 Database: {self.db.db_path}
🕒 Last Updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            self.stats_text.value = stats_text
            self.stats_text.update()
            
            conn.close()
            
        except Exception as e:
            if hasattr(self, 'stats_text'):
                self.stats_text.value = f"Error loading statistics: {e}"
                self.stats_text.update()

    def load_family_data(self, e):
        """Load family groups data directly to the family table"""
        print("🔧 DEBUG: load_family_data called")
        try:
            # Simply call load_surname_data to populate the family table
            self.load_surname_data()
            self.show_snack_bar(f"✅ Family data loaded successfully", ft.Colors.GREEN_600)
            
        except Exception as ex:
            print(f"🔧 DEBUG: Error in load_family_data: {ex}")
            self.show_snack_bar(f"❌ Error loading families: {ex}", ft.Colors.RED_600)

    def on_family_select(self, surname, voter_count=None):
        """Show insert panel for surname table instead of modal dialog"""
        try:
            # Ensure surname table exists
            self.create_surname_table()
            
            # If voter_count not provided, get it from election_c table
            if voter_count is None:
                conn = sqlite3.connect('demo_voters.db')
                cursor = conn.cursor()
                
                # Get voter count for this surname
                cursor.execute("""
                    SELECT COUNT(DISTINCT identity_no) 
                    FROM election_c 
                    WHERE surname = ? AND identity_no IS NOT NULL AND identity_no != '' AND identity_no != 'N/A'
                """, (surname,))
                
                voter_count = cursor.fetchone()[0]
                conn.close()
            
            # Show insert panel modal for surname table
            self.show_surname_insert_modal(surname, voter_count)
            
        except Exception as e:
            self.show_snack_bar(f"Error preparing surname insert: {e}", ft.Colors.RED_600)

    def create_surname_table(self):
        """Create surname table if it doesn't exist with UNIQUE constraint on surname"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='surname'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                # Create new table with UNIQUE constraint
                cursor.execute("""
                    CREATE TABLE surname (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        surname TEXT NOT NULL UNIQUE,
                        number_voter INTEGER DEFAULT 0,
                        note1 TEXT DEFAULT '',
                        note2 TEXT DEFAULT '',
                        note3 TEXT DEFAULT ''
                    )
                """)
            else:
                # Check if UNIQUE constraint exists, if not recreate table
                cursor.execute("PRAGMA table_info(surname)")
                columns = cursor.fetchall()
                
                # Check if surname has UNIQUE constraint by trying to create index
                try:
                    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_surname_unique ON surname(surname)")
                except:
                    pass  # Index might already exist
            
            conn.commit()
            conn.close()
            
        except Exception as ex:
            self.show_snack_bar(f"Error creating surname table: {ex}", ft.Colors.RED_600)

    def show_surname_insert_modal(self, surname, voter_count):
        """Show insert/update form modal for surname table"""
        print(f"🔧 DEBUG: Creating surname insert/update modal for: {surname}")
        
        # Check if surname already exists and get existing notes
        existing_notes = {"note1": "", "note2": "", "note3": ""}
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            cursor.execute("SELECT note1, note2, note3 FROM surname WHERE surname = ?", (surname,))
            existing_record = cursor.fetchone()
            if existing_record:
                existing_notes["note1"] = existing_record[0] or ""
                existing_notes["note2"] = existing_record[1] or ""
                existing_notes["note3"] = existing_record[2] or ""
            conn.close()
        except:
            pass  # If table doesn't exist yet, use empty notes
        
        # Drag state variables
        self.surname_modal_is_dragging = False
        self.surname_modal_drag_start_x = 0
        self.surname_modal_drag_start_y = 0
        self.surname_modal_start_left = (1400-400)//2
        self.surname_modal_start_top = 50
        
        def close_modal(e):
            print("🔧 DEBUG: Closing surname insert modal")
            if hasattr(self, 'surname_insert_modal') and self.surname_insert_modal in self.page.overlay:
                self.page.overlay.remove(self.surname_insert_modal)
                self.page.update()
        
        def on_surname_pan_start(e):
            """Start dragging the surname modal"""
            self.surname_modal_is_dragging = True
            self.surname_modal_drag_start_x = e.global_x
            self.surname_modal_drag_start_y = e.global_y
            print(f"🔧 DEBUG: Surname modal drag started at ({e.global_x}, {e.global_y})")
        
        def on_surname_pan_update(e):
            """Update surname modal position during drag"""
            if self.surname_modal_is_dragging:
                # Calculate new position
                delta_x = e.global_x - self.surname_modal_drag_start_x
                delta_y = e.global_y - self.surname_modal_drag_start_y
                
                new_left = max(0, min(1400-400, self.surname_modal_start_left + delta_x))
                new_top = max(0, min(800-600, self.surname_modal_start_top + delta_y))
                
                # Update modal position
                self.surname_insert_modal.left = new_left
                self.surname_insert_modal.top = new_top
                self.surname_insert_modal.update()
        
        def on_surname_pan_end(e):
            """End dragging and save new position"""
            if self.surname_modal_is_dragging:
                self.surname_modal_is_dragging = False
                self.surname_modal_start_left = self.surname_insert_modal.left
                self.surname_modal_start_top = self.surname_insert_modal.top
                print(f"🔧 DEBUG: Surname modal drag ended at ({self.surname_modal_start_left}, {self.surname_modal_start_top})")
        
        def save_surname_record(e):
            try:
                # Get values from form fields
                surname_value = self.surname_field.value or surname
                number_voter_value = int(self.number_voter_field.value) if self.number_voter_field.value else voter_count
                note1_value = self.note1_field.value or ""
                note2_value = self.note2_field.value or ""
                note3_value = self.note3_field.value or ""
                
                # Insert or update surname table (prevent duplicates)
                conn = sqlite3.connect('demo_voters.db')
                cursor = conn.cursor()
                
                # Check if surname already exists
                cursor.execute("SELECT id FROM surname WHERE surname = ?", (surname_value,))
                existing_record = cursor.fetchone()
                
                if existing_record:
                    # Update existing record
                    cursor.execute("""
                        UPDATE surname 
                        SET number_voter = ?, note1 = ?, note2 = ?, note3 = ?
                        WHERE surname = ?
                    """, (number_voter_value, note1_value, note2_value, note3_value, surname_value))
                    action = "อัปเดต"
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO surname (surname, number_voter, note1, note2, note3)
                        VALUES (?, ?, ?, ?, ?)
                    """, (surname_value, number_voter_value, note1_value, note2_value, note3_value))
                    action = "บันทึก"
                
                conn.commit()
                conn.close()
                
                self.show_snack_bar(f"✅ {action}ข้อมูลสำเร็จ", ft.Colors.GREEN_600)
                close_modal(e)
                # รีเฟรชสถิติ the surname table data
                self.load_surname_data()
                
            except Exception as ex:
                self.show_snack_bar(f"❌ Error saving: {ex}", ft.Colors.RED_600)
        
        def delete_surname_record(e):
            """Delete surname record from database"""
            try:
                conn = sqlite3.connect('demo_voters.db')
                cursor = conn.cursor()
                
                # Check if record exists before deleting
                cursor.execute("SELECT id FROM surname WHERE surname = ?", (surname,))
                existing_record = cursor.fetchone()
                
                if existing_record:
                    cursor.execute("DELETE FROM surname WHERE surname = ?", (surname,))
                    conn.commit()
                    conn.close()
                    
                    self.show_snack_bar(f"✅ ลบข้อมูล {surname} สำเร็จ", ft.Colors.GREEN_600)
                    close_modal(e)
                    # รีเฟรชสถิติ the surname table data
                    self.load_surname_data()
                else:
                    conn.close()
                    self.show_snack_bar("❌ ไม่พบข้อมูลที่จะลบ", ft.Colors.RED_600)
                    
            except Exception as ex:
                self.show_snack_bar(f"❌ Error deleting: {ex}", ft.Colors.RED_600)
        
        # Create form fields
        self.surname_field = ft.TextField(
            label="นามสกุล", 
            value=surname, 
            width=300,
            border_color=ft.Colors.BLUE_400,
            read_only=True
        )
        
        self.number_voter_field = ft.TextField(
            label="จำนวนผู้มีสิทธิ์เลือกตั้ง", 
            value=str(voter_count), 
            width=300,
            border_color=ft.Colors.BLUE_400
        )
        
        self.note1_field = ft.TextField(
            label="หมายเหตุ 1", 
            value=existing_notes["note1"], 
            width=300,
            border_color=ft.Colors.BLUE_400
        )
        
        self.note2_field = ft.TextField(
            label="หมายเหตุ 2", 
            value=existing_notes["note2"], 
            width=300,
            border_color=ft.Colors.BLUE_400
        )
        
        self.note3_field = ft.TextField(
            label="หมายเหตุ 3", 
            value=existing_notes["note3"], 
            width=300,
            multiline=True,
            min_lines=2,
            max_lines=3,
            border_color=ft.Colors.BLUE_400
        )
        
        # Determine if this is insert or update
        is_update = any(existing_notes.values())
        operation = "แก้ไข" if is_update else "เพิ่ม"
        operation_thai = "แก้ไข" if is_update else "เพิ่ม"
        
        # Create form in ListView
        form_items = [
            ft.ListTile(
                title=ft.Text(f"📝 {operation} Surname Record", size=16, weight=ft.FontWeight.BOLD),
                subtitle=ft.Text(f"{operation_thai}ข้อมูลสำหรับ: {surname}", size=12, color=ft.Colors.BLUE_600)
            ),
            ft.Divider(),
            ft.ListTile(title=self.surname_field),
            ft.ListTile(title=self.number_voter_field),
            ft.ListTile(title=self.note1_field),
            ft.ListTile(title=self.note2_field),
            ft.ListTile(title=self.note3_field),
            ft.Divider(),
            ft.ListTile(
                title=ft.Row([
                    ft.ElevatedButton(
                        "💾 บันทึก",
                        icon=ft.Icons.SAVE,
                        on_click=save_surname_record,
                        bgcolor=ft.Colors.GREEN_600,
                        color=ft.Colors.WHITE,
                        expand=True
                    ),
                    ft.Container(width=5),
                    ft.ElevatedButton(
                        "🗑️ ลบ",
                        icon=ft.Icons.DELETE,
                        on_click=delete_surname_record,
                        bgcolor=ft.Colors.ORANGE_600,
                        color=ft.Colors.WHITE,
                        expand=True
                    ),
                    ft.Container(width=5),
                    ft.ElevatedButton(
                        "❌ ยกเลิก",
                        icon=ft.Icons.CANCEL,
                        on_click=close_modal,
                        bgcolor=ft.Colors.RED_600,
                        color=ft.Colors.WHITE,
                        expand=True
                    )
                ])
            )
        ]
        
        # Create modal overlay
        self.surname_insert_modal = ft.Container(
            content=ft.Column([
                # Modal header with drag functionality
                ft.GestureDetector(
                    content=ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.DRAG_INDICATOR, color=ft.Colors.WHITE, size=16),
                            ft.Text("📝 เพิ่มข้อมูลนามสกุล", size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE, expand=True),
                            ft.IconButton(
                                ft.Icons.CLOSE,
                                icon_color=ft.Colors.WHITE,
                                on_click=close_modal
                            )
                        ]),
                        bgcolor=ft.Colors.GREEN_800,
                        padding=ft.padding.all(15),
                        border_radius=ft.border_radius.only(top_left=10, top_right=10)
                    ),
                    on_pan_start=on_surname_pan_start,
                    on_pan_update=on_surname_pan_update,
                    on_pan_end=on_surname_pan_end
                ),
                # Modal content with form
                ft.Container(
                    content=ft.ListView(
                        controls=form_items,
                        expand=True,
                        spacing=5
                    ),
                    height=520,
                    padding=ft.padding.all(10),
                    bgcolor=ft.Colors.WHITE
                )
            ]),
            width=400,
            height=600,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.BLACK26
            ),
            margin=ft.margin.all(20),
            top=50,
            left=(1400-400)//2,  # Center horizontally for desktop
            right=(1400-400)//2
        )
        
        # Add to page overlay
        self.page.overlay.append(self.surname_insert_modal)
        self.page.update()
        print(f"🔧 DEBUG: Surname insert modal overlay added to page")

    def save_family_data_to_db(self, e):
        """Save current family data to family_voter table"""
        try:
            min_voters = 2  # Default minimum voters
            
            # Connect to demo_voters.db and query election_c table
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Count surnames and unique houses from election_c table
            cursor.execute("""
                SELECT surname, COUNT(DISTINCT identity_no) as total_voters,
                       COUNT(DISTINCT no_address || '|' || mo_address) as unique_houses
                FROM election_c 
                WHERE surname IS NOT NULL AND surname != '' AND surname != 'N/A'
                  AND identity_no IS NOT NULL AND identity_no != '' AND identity_no != 'N/A'
                GROUP BY surname 
                HAVING COUNT(DISTINCT identity_no) >= ?
                ORDER BY total_voters DESC, surname ASC
            """, (min_voters,))
            
            families = cursor.fetchall()
            conn.close()
            
            # Save data to family_voter table
            saved_count = 0
            for surname, total_voters, unique_houses in families:
                self.db.save_family_voter_data(surname, total_voters, unique_houses)
                saved_count += 1
            
            self.show_snack_bar(f"✅ Successfully saved {saved_count} family records to family_voter table", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error saving family data: {ex}", ft.Colors.RED_600)

    def load_house_data(self, e):
        """Load house analysis data from election_c table"""
        try:
            min_voters = int(self.house_min_voters.value) if self.house_min_voters.value else 2
            
            # Connect to demo_voters.db and query election_c table
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Group by address and mo_address, count unique people, and get family names
            cursor.execute("""
                SELECT no_address, mo_address, COUNT(DISTINCT identity_no) as people_count,
                       GROUP_CONCAT(DISTINCT surname) as family_names
                FROM election_c 
                WHERE no_address IS NOT NULL AND no_address != '' AND no_address != 'N/A'
                  AND identity_no IS NOT NULL AND identity_no != '' AND identity_no != 'N/A'
                GROUP BY no_address, mo_address 
                HAVING COUNT(DISTINCT identity_no) >= ?
                ORDER BY people_count DESC, no_address ASC
            """, (min_voters,))
            
            addresses = cursor.fetchall()
            conn.close()
            
            self.house_table.rows.clear()
            
            for address_no, mo_address, people_count, family_names in addresses:
                # Limit family names display to avoid too long text
                family_display = family_names[:50] + "..." if family_names and len(family_names) > 50 else family_names
                
                # Create Details button
                details_button = ft.ElevatedButton(
                    "👁️",
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREEN_600,
                    width=50,
                    height=30,
                    on_click=lambda e, addr=address_no, mo=mo_address: self.on_house_select(addr, mo)
                )
                
                self.house_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(address_no or "N/A"))),
                            ft.DataCell(ft.Text(str(mo_address or "N/A"))),
                            ft.DataCell(ft.Text(str(people_count))),
                            ft.DataCell(ft.Text(str(family_display or "N/A"))),
                            ft.DataCell(ft.Text("unknown")),  # Support level - default
                            ft.DataCell(details_button)
                        ]
                    )
                )
            
            self.house_table.update()
            self.show_snack_bar(f"📋 Loaded {len(addresses)} addresses from election_c (sorted by people count)", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading addresses: {ex}", ft.Colors.RED_600)

    def on_house_select(self, address_no, mo_address):
        """Handle address selection from election_c table"""
        try:
            # Connect to demo_voters.db and query election_c table
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get all unique people at this address (by identity number)
            cursor.execute("""
                SELECT identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE no_address = ? AND mo_address = ?
                  AND identity_no IS NOT NULL AND identity_no != '' AND identity_no != 'N/A'
                GROUP BY identity_no
                ORDER BY surname, name
            """, (address_no, mo_address))
            
            residents = cursor.fetchall()
            conn.close()
            
            details_text = f"🏠 Address: {address_no or 'N/A'}, หมู่ที่: {mo_address or 'N/A'}\n"
            details_text += f"📊 Total Residents: {len(residents)}\n\n"
            details_text += "👥 People at this Address:\n"
            details_text += "─" * 80 + "\n"
            
            for resident in residents:
                identity_no, title, name, surname, addr_no, mo_addr, day_birth, month, year, sex, remark, birthday = resident
                details_text += f"• {title or ''} {name or ''} {surname or ''}\n"
                details_text += f"  🆔 ID: {identity_no or 'N/A'}, 👤 Gender: {sex or 'N/A'}\n"
                details_text += f"  🎂 Birth: {day_birth or ''}/{month or ''}/{year or ''}\n"
                if remark:
                    details_text += f"  📝 Remark: {remark}\n"
                details_text += "\n"
            
            # Show house details in modal dialog instead of panel  
            self.show_desktop_modal_dialog(f"🏠 รายละเอียดบ้าน: {address_no}, หมู่ {mo_address}", details_text, 600, 700)
            
        except Exception as e:
            self.show_snack_bar(f"Error loading address details: {e}", ft.Colors.RED_600)

    def load_birthplace_data(self, e):
        """Load birthplace data from election_c and id_birthplace tables"""
        try:
            min_voters = 1  # Default minimum voters for automatic loading
            
            # Connect to demo_voters.db (election_c table)
            conn1 = sqlite3.connect('demo_voters.db')
            cursor1 = conn1.cursor()
            
            # Get all identity numbers and extract birth codes (position 2-5, 4 digits)
            cursor1.execute("SELECT identity_no FROM election_c WHERE identity_no IS NOT NULL AND identity_no != ''")
            identity_numbers = cursor1.fetchall()
            conn1.close()
            
            # Extract birth codes and count them
            birth_code_counts = {}
            for (identity_no,) in identity_numbers:
                try:
                    # Extract 4-digit birth code from position 2-5 (after first dash)
                    parts = str(identity_no).split('-')
                    if len(parts) >= 2 and len(parts[1]) >= 4:
                        birth_code = int(parts[1][:4])  # First 4 digits after first dash
                        birth_code_counts[birth_code] = birth_code_counts.get(birth_code, 0) + 1
                except (ValueError, IndexError):
                    continue
            
            # Connect to demo_geo_voters.db (id_birthplace table)
            conn2 = sqlite3.connect('demo_geo_voters.db')
            cursor2 = conn2.cursor()
            
            # Get birthplace information for codes that have enough people
            self.birthplace_table.rows.clear()
            birthplace_data = []
            
            for birth_code, count in birth_code_counts.items():
                if count >= min_voters:
                    # Get district and province from id_birthplace table
                    cursor2.execute("SELECT District, Province FROM id_birthplace WHERE Code = ?", (birth_code,))
                    result = cursor2.fetchone()
                    
                    if result:
                        district, province = result
                        birthplace_data.append((birth_code, district, province, count))
                    else:
                        # If no match found, show with N/A
                        birthplace_data.append((birth_code, "N/A", "N/A", count))
            
            conn2.close()
            
            # Sort by people count (descending)
            birthplace_data.sort(key=lambda x: x[3], reverse=True)
            
            # Populate table
            for birth_code, district, province, count in birthplace_data:
                # Create Details button
                details_button = ft.ElevatedButton(
                    "👁️",
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.ORANGE_600,
                    width=50,
                    height=30,
                    on_click=lambda e, code=birth_code: self.on_birthplace_select(code)
                )
                
                self.birthplace_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(birth_code))),
                            ft.DataCell(ft.Text(str(district))),
                            ft.DataCell(ft.Text(str(province))),
                            ft.DataCell(ft.Text(str(count))),
                            ft.DataCell(ft.Text("unknown")),  # Support level - default
                            ft.DataCell(details_button)
                        ]
                    )
                )
            
            self.birthplace_table.update()
            self.show_snack_bar(f"📋 Loaded {len(birthplace_data)} birth places from election_c + id_birthplace (sorted by count)", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading birthplaces: {ex}", ft.Colors.RED_600)

    def on_birthplace_select(self, birth_code):
        """Handle birthplace selection from election_c table"""
        try:
            # Connect to demo_voters.db (election_c table)
            conn1 = sqlite3.connect('demo_voters.db')
            cursor1 = conn1.cursor()
            
            # Get all people with this birth code in their identity number
            cursor1.execute("""
                SELECT identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no LIKE ?
                ORDER BY surname, name
            """, (f'%-{birth_code:04d}-%',))
            
            people = cursor1.fetchall()
            conn1.close()
            
            # Get district and province info from id_birthplace
            conn2 = sqlite3.connect('demo_geo_voters.db')
            cursor2 = conn2.cursor()
            cursor2.execute("SELECT District, Province FROM id_birthplace WHERE Code = ?", (birth_code,))
            location_info = cursor2.fetchone()
            conn2.close()
            
            district = location_info[0] if location_info else "N/A"
            province = location_info[1] if location_info else "N/A"
            
            details_text = f"🌍 Birth Code: {birth_code}\n"
            details_text += f"📍 District: {district}, Province: {province}\n"
            details_text += f"📊 Total People: {len(people)}\n\n"
            details_text += "👥 People born in this region:\n"
            details_text += "─" * 80 + "\n"
            
            for person in people:
                identity_no, title, name, surname, addr_no, mo_addr, day_birth, month, year, sex, remark, birthday = person
                details_text += f"• {title or ''} {name or ''} {surname or ''}\n"
                details_text += f"  🆔 ID: {identity_no or 'N/A'}, 👤 Gender: {sex or 'N/A'}\n"
                details_text += f"  🏠 Address: {addr_no or 'N/A'}, หมู่: {mo_addr or 'N/A'}\n"
                details_text += f"  🎂 Birth: {day_birth or ''}/{month or ''}/{year or ''}\n"
                if remark:
                    details_text += f"  📝 Remark: {remark}\n"
                details_text += "\n"
            
            # Show birthplace details in modal dialog instead of panel
            self.show_desktop_modal_dialog(f"🌍 รายละเอียดสถานที่เกิด: {district}, {province}", details_text, 600, 700)
            
        except Exception as e:
            self.show_snack_bar(f"Error loading birthplace details: {e}", ft.Colors.RED_600)

    def add_campaign_activity(self, e):
        """Add a new campaign activity"""
        try:
            activity_type = self.activity_type.value
            target_group = self.target_group.value
            target_id = self.target_id.value
            description = self.activity_description.value
            
            if not all([activity_type, target_group, target_id, description]):
                self.show_snack_bar("Please fill all fields", ft.Colors.ORANGE_600)
                return
            
            self.db.add_campaign_activity(activity_type, target_group, target_id, description)
            
            # Clear form
            self.target_id.value = ""
            self.activity_description.value = ""
            self.target_id.update()
            self.activity_description.update()
            
            self.load_campaign_activities(None)
            self.show_snack_bar("Campaign activity added successfully", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error adding activity: {ex}", ft.Colors.RED_600)

    def load_campaign_activities(self, e):
        """Load campaign activities"""
        try:
            activities = self.db.get_campaign_activities(100)
            
            self.activities_table.rows.clear()
            
            for activity in activities:
                self.activities_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(activity['activity_date']))),
                            ft.DataCell(ft.Text(str(activity['activity_type']))),
                            ft.DataCell(ft.Text(str(activity['target_group']))),
                            ft.DataCell(ft.Text(str(activity['target_id']))),
                            ft.DataCell(ft.Text(str(activity['description']))),
                            ft.DataCell(ft.Text(str(activity['result'] or "")))
                        ]
                    )
                )
            
            self.activities_table.update()
            self.show_snack_bar(f"📋 Loaded {len(activities)} activities", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading activities: {ex}", ft.Colors.RED_600)

    def generate_family_chart(self, e):
        """Generate and display family chart using matplotlib"""
        try:
            # Get family data
            families = self.db.get_family_groups(2)
            if not families:
                self.show_snack_bar("❌ No family data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data (top 15 families)
            top_families = families[:15]
            surnames = [f['surname'] for f in top_families]
            voter_counts = [f['total_voters'] for f in top_families]
            
            # Create figure with font that supports English
            plt.figure(figsize=(12, 8))
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            bars = plt.bar(range(len(surnames)), voter_counts, color='steelblue', alpha=0.8)
            
            # Customize chart
            plt.xlabel('Family (Surname)', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Voters', fontsize=12, fontweight='bold')
            plt.title('Top 15 Families by Voter Count', fontsize=16, fontweight='bold', pad=20)
            plt.xticks(range(len(surnames)), surnames, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, voter_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            # Display in Flet
            self.chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.chart_container.update()
            
            plt.close()  # Clean up
            self.show_snack_bar(f"📊 Family chart generated for {len(top_families)} families", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating family chart: {ex}", ft.Colors.RED_600)

    def generate_house_chart(self, e):
        """Generate and display house chart using matplotlib"""
        try:
            # Get house data
            # Get house data directly from election_c table
            import sqlite3
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get house statistics from election_c table
            cursor.execute("""
                SELECT 
                    no_address || '/' || mo_address as house_id,
                    COUNT(*) as voter_count,
                    GROUP_CONCAT(DISTINCT surname) as family_names
                FROM election_c 
                WHERE no_address IS NOT NULL AND no_address != '' 
                  AND mo_address IS NOT NULL AND mo_address != ''
                GROUP BY no_address, mo_address
                HAVING COUNT(*) >= 2
                ORDER BY voter_count DESC
                LIMIT 20
            """)
            
            houses_data = cursor.fetchall()
            conn.close()
            
            if not houses_data:
                self.show_snack_bar("❌ No house data available", ft.Colors.RED_600)
                return
            
            # Convert to the expected format
            houses = []
            for house_id, voter_count, family_names in houses_data:
                houses.append({
                    'house_id': house_id,
                    'voter_count': voter_count,
                    'family_names': family_names
                })
            if not houses:
                self.show_snack_bar("❌ No house data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data (top 20 houses)
            top_houses = houses[:20]
            house_ids = [h['house_id'] for h in top_houses]
            voter_counts = [h['voter_count'] for h in top_houses]
            
            # Create figure with font that supports English
            plt.figure(figsize=(12, 8))
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            bars = plt.bar(range(len(house_ids)), voter_counts, color='darkorange', alpha=0.8)
            
            # Customize chart
            plt.xlabel('House ID', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Voters', fontsize=12, fontweight='bold')
            plt.title('Top 20 Houses by Voter Count', fontsize=16, fontweight='bold', pad=20)
            plt.xticks(range(len(house_ids)), house_ids, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, voter_counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            # Display in Flet
            self.chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.chart_container.update()
            
            plt.close()  # Clean up
            self.show_snack_bar(f"📊 House chart generated for {len(top_houses)} houses", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating house chart: {ex}", ft.Colors.RED_600)

    def generate_stats_chart(self, e):
        """Generate and display statistics summary chart"""
        try:
            # Get database statistics
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM voters")
            total_voters = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT surname) FROM voters")
            total_families = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT house_id) FROM voters")
            total_houses = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT birthplace_code) FROM voters")
            total_birthplaces = cursor.fetchone()[0]
            
            conn.close()
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data
            categories = ['Total Voters', 'Families', 'Houses', 'Birthplaces']
            values = [total_voters, total_families, total_houses, total_birthplaces]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
            
            # Create figure with pie chart and font setting
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Pie chart
            ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax1.set_title('Database Distribution', fontsize=14, fontweight='bold', pad=20)
            
            # Bar chart
            bars = ax2.bar(categories, values, color=colors, alpha=0.8)
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title('Database Statistics', fontsize=14, fontweight='bold', pad=20)
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:,}', ha='center', va='bottom', fontweight='bold')
            
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            # Display in Flet
            self.chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.chart_container.update()
            
            plt.close()  # Clean up
            self.show_snack_bar(f"📊 Statistics chart generated: {total_voters:,} voters total", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating statistics chart: {ex}", ft.Colors.RED_600)

    def export_report(self, e):
        """Export comprehensive report"""
        try:
            # Generate timestamp for filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"voter_analysis_report_{timestamp}.xlsx"
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Family analysis
                families = self.db.get_family_groups(1)
                df_families = pd.DataFrame(families)
                df_families.to_excel(writer, sheet_name='Family_Analysis', index=False)
                
                # House analysis
                houses = self.db.get_house_analysis(1)
                df_houses = pd.DataFrame(houses)
                df_houses.to_excel(writer, sheet_name='House_Analysis', index=False)
                
                # Birthplace analysis
                birthplaces = self.db.get_birthplace_groups(1)
                df_birthplaces = pd.DataFrame(birthplaces)
                df_birthplaces.to_excel(writer, sheet_name='Birthplace_Analysis', index=False)
                
                # Campaign activities
                activities = self.db.get_campaign_activities(500)
                df_activities = pd.DataFrame(activities)
                df_activities.to_excel(writer, sheet_name='Campaign_Activities', index=False)
            
            self.show_snack_bar(f"📄 Report exported successfully to: {file_path}", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error exporting report: {ex}", ft.Colors.RED_600)

    # IDENTITY CARD CHART FUNCTIONS FOR TAB 11
    def generate_gender_chart(self, e):
        """Generate and display gender distribution chart from election_c table"""
        try:
            # Get gender data from election_c
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sex, COUNT(*) as count 
                FROM election_c 
                WHERE sex IS NOT NULL AND sex != '' 
                GROUP BY sex 
                ORDER BY count DESC
            """)
            gender_data = cursor.fetchall()
            conn.close()
            
            if not gender_data:
                self.show_snack_bar("❌ No gender data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data
            genders = [row[0] for row in gender_data]
            counts = [row[1] for row in gender_data]
            colors = ['#FF69B4', '#4169E1', '#32CD32', '#FFD700']
            
            # Create figure with pie chart and bar chart
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # Pie chart
            ax1.pie(counts, labels=genders, colors=colors[:len(genders)], autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax1.set_title('Gender Distribution (Pie Chart)', fontsize=14, fontweight='bold', pad=20)
            
            # Bar chart
            bars = ax2.bar(genders, counts, color=colors[:len(genders)], alpha=0.8)
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax2.set_title('Gender Distribution (Bar Chart)', fontsize=14, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save to bytes and display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            self.id_chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.id_chart_container.update()
            
            plt.close()
            total_people = sum(counts)
            self.show_snack_bar(f"📊 Gender chart generated: {total_people:,} people analyzed", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating gender chart: {ex}", ft.Colors.RED_600)

    def generate_birth_year_chart(self, e):
        """Generate and display birth year analysis chart from election_c table"""
        try:
            # Get birth year data from election_c (Buddhist Era years)
            import sqlite3
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # First, let's check what columns are available
            cursor.execute("PRAGMA table_info(election_c)")
            columns = cursor.fetchall()
            print(f"🔧 DEBUG: Available columns in election_c: {[col[1] for col in columns]}")
            
            # Try different possible year column names
            year_columns = ['year', 'year_birth', 'birth_year', 'year_birthday', 'birthday_year']
            year_data = []
            
            for col_name in year_columns:
                try:
                    cursor.execute(f"""
                        SELECT {col_name}, COUNT(*) as count 
                        FROM election_c 
                        WHERE {col_name} IS NOT NULL AND {col_name} != '' 
                          AND CAST({col_name} AS INTEGER) BETWEEN 2400 AND 2600
                        GROUP BY {col_name} 
                        ORDER BY {col_name}
                    """)
                    temp_data = cursor.fetchall()
                    if temp_data:
                        year_data = temp_data
                        print(f"🔧 DEBUG: Found data using column '{col_name}': {len(year_data)} records")
                        break
                except Exception as e:
                    print(f"🔧 DEBUG: Column '{col_name}' not available or error: {e}")
                    continue
            
            conn.close()
            
            if not year_data:
                self.show_snack_bar("❌ No birth year data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data - Convert Buddhist Era to Common Era for display
            # Handle decimal values in year data
            be_years = []
            counts = []
            for row in year_data:
                try:
                    # Convert string to float first, then to int
                    be_year = int(float(row[0]))
                    be_years.append(be_year)
                    counts.append(row[1])
                except (ValueError, TypeError) as e:
                    print(f"🔧 DEBUG: Skipping invalid year data: {row[0]} - {e}")
                    continue
            
            if not be_years:
                self.show_snack_bar("❌ No valid birth year data found", ft.Colors.RED_600)
                return
            counts = [row[1] for row in year_data]
            ce_years = [be_year - 543 for be_year in be_years]  # Convert BE to CE
            
            # Use CE years for display but show both in labels
            years = ce_years
            
            # Create figure
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.figure(figsize=(14, 8))
            
            # Line chart with markers
            plt.plot(years, counts, marker='o', linewidth=2, markersize=6, color='#8B4513', alpha=0.8)
            plt.fill_between(years, counts, alpha=0.3, color='#DEB887')
            
            plt.xlabel('Birth Year (Common Era)', fontsize=12, fontweight='bold')
            plt.ylabel('Number of People', fontsize=12, fontweight='bold')
            plt.title('Birth Year Distribution Analysis (Buddhist Era → Common Era)', fontsize=16, fontweight='bold', pad=20)
            plt.grid(True, alpha=0.3)
            
            # Highlight some points
            for i in range(0, len(years), max(1, len(years)//10)):
                plt.annotate(f'{counts[i]}', (years[i], counts[i]), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')
            
            plt.tight_layout()
            
            # Save to bytes and display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            self.id_chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.id_chart_container.update()
            
            plt.close()
            total_records = sum(counts)
            ce_year_range = f"{min(years)}-{max(years)} CE"
            be_year_range = f"{min(be_years)}-{max(be_years)} BE"
            self.show_snack_bar(f"📊 Birth year chart generated: {total_records:,} records ({ce_year_range}, {be_year_range})", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating birth year chart: {ex}", ft.Colors.RED_600)

    def generate_address_chart(self, e):
        """Generate and display address distribution chart from election_c table"""
        try:
            # Get address data from election_c
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT mo_address, COUNT(*) as count 
                FROM election_c 
                WHERE mo_address IS NOT NULL AND mo_address != '' 
                GROUP BY mo_address 
                ORDER BY count DESC 
                LIMIT 15
            """)
            address_data = cursor.fetchall()
            conn.close()
            
            if not address_data:
                self.show_snack_bar("❌ No address data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data
            addresses = [row[0] for row in address_data]
            counts = [row[1] for row in address_data]
            
            # Create figure
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.figure(figsize=(12, 8))
            
            # Horizontal bar chart for better readability
            bars = plt.barh(range(len(addresses)), counts, color='#4682B4', alpha=0.8)
            
            plt.xlabel('Number of People', fontsize=12, fontweight='bold')
            plt.ylabel('Address (หมู่ที่)', fontsize=12, fontweight='bold')
            plt.title('Top 15 Address Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.yticks(range(len(addresses)), addresses)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                        f'{count:,}', ha='left', va='center', fontweight='bold')
            
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save to bytes and display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            self.id_chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.id_chart_container.update()
            
            plt.close()
            total_addresses = sum(counts)
            self.show_snack_bar(f"📊 Address chart generated: {total_addresses:,} people in top 15 addresses", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating address chart: {ex}", ft.Colors.RED_600)

    def generate_monthly_births_chart(self, e):
        """Generate and display monthly birth distribution chart from election_c table"""
        try:
            # Get monthly birth data from election_c (Thai month abbreviations)
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT month, COUNT(*) as count 
                FROM election_c 
                WHERE month IS NOT NULL AND month != ''
                GROUP BY month 
                ORDER BY month
            """)
            month_data = cursor.fetchall()
            conn.close()
            
            if not month_data:
                self.show_snack_bar("❌ No birth month data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Thai month mapping to English and month numbers
            thai_month_map = {
                'ม.ค.': (1, 'Jan'), 'ม.ค': (1, 'Jan'),      # January
                'ก.พ.': (2, 'Feb'),                          # February  
                'มี.ค.': (3, 'Mar'),                         # March
                'เม.ย.': (4, 'Apr'),                        # April
                'พ.ค.': (5, 'May'), 'พ.ค': (5, 'May'),      # May
                'มิ.ย.': (6, 'Jun'), 'มิ.ย': (6, 'Jun'),     # June
                'ก.ค.': (7, 'Jul'),                          # July
                'ส.ค.': (8, 'Aug'), 'ส.ค': (8, 'Aug'),      # August
                'ก.ย.': (9, 'Sep'),                          # September
                'ต.ค.': (10, 'Oct'),                         # October
                'พ.ย.': (11, 'Nov'),                         # November
                'ธ.ค.': (12, 'Dec')                          # December
            }
            
            # Prepare data with month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            # Create complete data for all 12 months
            month_counts = [0] * 12
            for thai_month, count in month_data:
                if thai_month in thai_month_map:
                    month_num, _ = thai_month_map[thai_month]
                    month_counts[month_num-1] += count  # Use += to handle duplicates like 'พ.ค.' and 'พ.ค'
            
            # Create figure
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.figure(figsize=(12, 8))
            
            # Bar chart with gradient colors
            colors = plt.cm.viridis(range(12))
            bars = plt.bar(month_names, month_counts, color=colors, alpha=0.8)
            
            plt.xlabel('Birth Month', fontsize=12, fontweight='bold')
            plt.ylabel('Number of People', fontsize=12, fontweight='bold')
            plt.title('Monthly Birth Distribution (Thai Months → English)', fontsize=16, fontweight='bold', pad=20)
            
            # Add value labels on bars
            for bar, count in zip(bars, month_counts):
                if count > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(month_counts)*0.01,
                            f'{count}', ha='center', va='bottom', fontweight='bold')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save to bytes and display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            self.id_chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.id_chart_container.update()
            
            plt.close()
            total_births = sum(month_counts)
            processed_months = len([c for c in month_counts if c > 0])
            self.show_snack_bar(f"📊 Monthly births chart generated: {total_births:,} people across {processed_months} months (Thai→English)", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating monthly births chart: {ex}", ft.Colors.RED_600)

    def generate_title_chart(self, e):
        """Generate and display title distribution chart from election_c table"""
        try:
            # Get title data from election_c
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT name_title, COUNT(*) as count 
                FROM election_c 
                WHERE name_title IS NOT NULL AND name_title != '' 
                GROUP BY name_title 
                ORDER BY count DESC
            """)
            title_data = cursor.fetchall()
            conn.close()
            
            if not title_data:
                self.show_snack_bar("❌ No title data available", ft.Colors.RED_600)
                return
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data
            titles = [row[0] for row in title_data]
            counts = [row[1] for row in title_data]
            colors = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#DDA0DD', '#FFA500']
            
            # Create figure
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            plt.figure(figsize=(12, 8))
            
            # Pie chart
            plt.pie(counts, labels=titles, colors=colors[:len(titles)], autopct='%1.1f%%', 
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            plt.title('Title Distribution', fontsize=16, fontweight='bold', pad=20)
            
            plt.tight_layout()
            
            # Save to bytes and display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            self.id_chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.id_chart_container.update()
            
            plt.close()
            total_people = sum(counts)
            self.show_snack_bar(f"📊 Title chart generated: {total_people:,} people with {len(titles)} different titles", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating title chart: {ex}", ft.Colors.RED_600)

    def generate_identity_stats_chart(self, e):
        """Generate and display comprehensive identity statistics chart from election_c table"""
        try:
            # Get comprehensive statistics from election_c
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get various statistics
            cursor.execute("SELECT COUNT(id) FROM election_c")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT identity_no) FROM election_c WHERE identity_no IS NOT NULL")
            unique_ids = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(id) FROM election_c WHERE sex IS NOT NULL AND sex != ''")
            records_with_gender = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(id) FROM election_c WHERE year IS NOT NULL AND year != ''")
            records_with_birth_year = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(id) FROM election_c WHERE mo_address IS NOT NULL AND mo_address != ''")
            records_with_address = cursor.fetchone()[0]
            
            conn.close()
            
            # Create matplotlib chart
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Prepare data
            categories = ['Total Records', 'Unique IDs', 'With Gender', 'With Birth Year', 'With Address']
            values = [total_records, unique_ids, records_with_gender, records_with_birth_year, records_with_address]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
            
            # Create figure with multiple charts
            plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Bar chart
            bars = ax1.bar(categories, values, color=colors, alpha=0.8)
            ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax1.set_title('Identity Database Statistics', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:,}', ha='center', va='bottom', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Pie chart for completeness
            completeness_values = [records_with_gender, records_with_birth_year, records_with_address]
            completeness_labels = ['Gender Data', 'Birth Year Data', 'Address Data']
            ax2.pie(completeness_values, labels=completeness_labels, colors=colors[2:5], autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
            ax2.set_title('Data Completeness', fontsize=14, fontweight='bold')
            
            # Data quality metrics
            quality_categories = ['Complete Records', 'Missing Gender', 'Missing Birth Year', 'Missing Address']
            quality_values = [
                min(records_with_gender, records_with_birth_year, records_with_address),
                total_records - records_with_gender,
                total_records - records_with_birth_year,
                total_records - records_with_address
            ]
            bars3 = ax3.bar(quality_categories, quality_values, color=['#2ECC71', '#E74C3C', '#F39C12', '#9B59B6'], alpha=0.8)
            ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax3.set_title('Data Quality Analysis', fontsize=14, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            for bar, value in zip(bars3, quality_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quality_values)*0.01,
                        f'{value:,}', ha='center', va='bottom', fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            
            # Summary text
            summary_text = f"""IDENTITY DATABASE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Total Records: {total_records:,}
🆔 Unique Identity Numbers: {unique_ids:,}
👥 Records with Gender: {records_with_gender:,}
📅 Records with Birth Year: {records_with_birth_year:,}
🏘️ Records with Address: {records_with_address:,}

📈 Data Completeness:
• Gender: {(records_with_gender/total_records*100):.1f}%
• Birth Year: {(records_with_birth_year/total_records*100):.1f}%
• Address: {(records_with_address/total_records*100):.1f}%

🎯 Data Quality Score: {(min(records_with_gender, records_with_birth_year, records_with_address)/total_records*100):.1f}%"""
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
            ax4.set_title('Database Summary', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save to bytes and display
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_data = img_buffer.getvalue()
            img_base64 = base64.b64encode(img_data).decode()
            
            self.id_chart_container.content = ft.Image(
                src_base64=img_base64,
                fit=ft.ImageFit.CONTAIN,
                width=900,
                height=480
            )
            self.id_chart_container.update()
            
            plt.close()
            self.show_snack_bar(f"📊 Identity statistics generated: {total_records:,} total records analyzed", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error generating identity statistics: {ex}", ft.Colors.RED_600)

    def export_identity_report(self, e):
        """Export comprehensive statistical identity report from election_c table"""
        try:
            # Generate timestamp for filename
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            file_path = f"voter_analysis_report_{timestamp}.xlsx"
            
            conn = sqlite3.connect('demo_voters.db')
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # ===== EXECUTIVE SUMMARY =====
                # Overall statistics
                total_records = pd.read_sql_query("SELECT COUNT(*) as total FROM election_c", conn).iloc[0]['total']
                
                summary_data = {
                    'Metric': [
                        'Total Voter Records',
                        'Male Voters',
                        'Female Voters',
                        'Average Age (calculated)',
                        'Most Common Title',
                        'Most Common Birth Year',
                        'Most Common Address',
                        'Records with Complete Data',
                        'Records with Missing Data'
                    ],
                    'Value': [
                        total_records,
                        pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE sex = 'ชาย'", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE sex = 'หญิง'", conn).iloc[0]['count'],
                        'Calculated from birth years',
                        pd.read_sql_query("SELECT name_title FROM election_c WHERE name_title IS NOT NULL GROUP BY name_title ORDER BY COUNT(*) DESC LIMIT 1", conn).iloc[0]['name_title'] if pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE name_title IS NOT NULL", conn).iloc[0]['count'] > 0 else 'N/A',
                        pd.read_sql_query("SELECT year FROM election_c WHERE year IS NOT NULL GROUP BY year ORDER BY COUNT(*) DESC LIMIT 1", conn).iloc[0]['year'] if pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE year IS NOT NULL", conn).iloc[0]['count'] > 0 else 'N/A',
                        pd.read_sql_query("SELECT mo_address FROM election_c WHERE mo_address IS NOT NULL GROUP BY mo_address ORDER BY COUNT(*) DESC LIMIT 1", conn).iloc[0]['mo_address'] if pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE mo_address IS NOT NULL", conn).iloc[0]['count'] > 0 else 'N/A',
                        pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE identity_no IS NOT NULL AND name IS NOT NULL AND surname IS NOT NULL", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(*) as count FROM election_c WHERE identity_no IS NULL OR name IS NULL OR surname IS NULL", conn).iloc[0]['count']
                    ]
                }
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Executive_Summary', index=False)
                
                # ===== COMPLETE DATA =====
                # All election_c data
                df_all = pd.read_sql_query("SELECT * FROM election_c ORDER BY order_no", conn)
                df_all.to_excel(writer, sheet_name='Complete_Voter_Data', index=False)
                
                # ===== GENDER ANALYSIS =====
                df_gender = pd.read_sql_query("""
                    SELECT 
                        sex as Gender,
                        COUNT(id) as Count,
                        ROUND(COUNT(id) * 100.0 / (SELECT COUNT(id) FROM election_c), 2) as Percentage
                    FROM election_c 
                    WHERE sex IS NOT NULL AND sex != '' 
                    GROUP BY sex 
                    ORDER BY Count DESC
                """, conn)
                df_gender.to_excel(writer, sheet_name='Gender_Analysis', index=False)
                
                # ===== BIRTH YEAR ANALYSIS =====
                df_year = pd.read_sql_query("""
                    SELECT 
                        year as Birth_Year,
                        COUNT(id) as Count,
                        ROUND(COUNT(id) * 100.0 / (SELECT COUNT(id) FROM election_c), 2) as Percentage
                    FROM election_c 
                    WHERE year IS NOT NULL AND year != '' 
                    GROUP BY year 
                    ORDER BY year DESC
                """, conn)
                df_year.to_excel(writer, sheet_name='Birth_Year_Analysis', index=False)
                
                # ===== ADDRESS ANALYSIS =====
                df_address = pd.read_sql_query("""
                    SELECT 
                        mo_address as Address_Number,
                        COUNT(id) as Count,
                        ROUND(COUNT(id) * 100.0 / (SELECT COUNT(id) FROM election_c), 2) as Percentage
                    FROM election_c 
                    WHERE mo_address IS NOT NULL AND mo_address != '' 
                    GROUP BY mo_address 
                    ORDER BY Count DESC
                """, conn)
                df_address.to_excel(writer, sheet_name='Address_Analysis', index=False)
                
                # ===== TITLE ANALYSIS =====
                df_title = pd.read_sql_query("""
                    SELECT 
                        name_title as Title,
                        COUNT(id) as Count,
                        ROUND(COUNT(id) * 100.0 / (SELECT COUNT(id) FROM election_c), 2) as Percentage
                    FROM election_c 
                    WHERE name_title IS NOT NULL AND name_title != '' 
                    GROUP BY name_title 
                    ORDER BY Count DESC
                """, conn)
                df_title.to_excel(writer, sheet_name='Title_Analysis', index=False)
                
                # ===== MONTHLY BIRTH ANALYSIS =====
                df_month = pd.read_sql_query("""
                    SELECT 
                        month as Birth_Month,
                        COUNT(id) as Count,
                        ROUND(COUNT(id) * 100.0 / (SELECT COUNT(id) FROM election_c), 2) as Percentage
                    FROM election_c 
                    WHERE month IS NOT NULL AND month != '' 
                    GROUP BY month 
                    ORDER BY 
                        CASE month
                            WHEN 'ม.ค.' THEN 1 WHEN 'ก.พ.' THEN 2 WHEN 'มี.ค.' THEN 3
                            WHEN 'เม.ย.' THEN 4 WHEN 'พ.ค.' THEN 5 WHEN 'มิ.ย.' THEN 6
                            WHEN 'ก.ค.' THEN 7 WHEN 'ส.ค.' THEN 8 WHEN 'ก.ย.' THEN 9
                            WHEN 'ต.ค.' THEN 10 WHEN 'พ.ย.' THEN 11 WHEN 'ธ.ค.' THEN 12
                            ELSE 13
                        END
                """, conn)
                df_month.to_excel(writer, sheet_name='Monthly_Birth_Analysis', index=False)
                
                # ===== DATA QUALITY ANALYSIS =====
                quality_data = {
                    'Field': ['Identity Number', 'Name', 'Surname', 'Title', 'Gender', 'Birth Year', 'Birth Month', 'Address', 'Complete Records'],
                    'Total Records': [total_records] * 9,
                    'Valid Records': [
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE identity_no IS NOT NULL AND identity_no != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE name IS NOT NULL AND name != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE surname IS NOT NULL AND surname != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE name_title IS NOT NULL AND name_title != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE sex IS NOT NULL AND sex != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE year IS NOT NULL AND year != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE month IS NOT NULL AND month != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE mo_address IS NOT NULL AND mo_address != ''", conn).iloc[0]['count'],
                        pd.read_sql_query("SELECT COUNT(id) as count FROM election_c WHERE identity_no IS NOT NULL AND name IS NOT NULL AND surname IS NOT NULL AND sex IS NOT NULL", conn).iloc[0]['count']
                    ]
                }
                
                df_quality = pd.DataFrame(quality_data)
                df_quality['Percentage'] = round(df_quality['Valid Records'] * 100.0 / df_quality['Total Records'], 2)
                df_quality.to_excel(writer, sheet_name='Data_Quality_Analysis', index=False)
                
                # ===== AGE GROUP ANALYSIS =====
                df_age_groups = pd.read_sql_query("""
                    SELECT 
                        CASE 
                            WHEN year >= 2500 THEN 'Under 25'
                            WHEN year >= 2480 THEN '25-45'
                            WHEN year >= 2460 THEN '46-65'
                            WHEN year >= 2440 THEN '66-85'
                            ELSE 'Over 85'
                        END as Age_Group,
                        COUNT(id) as Count,
                        ROUND(COUNT(id) * 100.0 / (SELECT COUNT(id) FROM election_c WHERE year IS NOT NULL), 2) as Percentage
                    FROM election_c 
                    WHERE year IS NOT NULL AND year != '' 
                    GROUP BY 
                        CASE 
                            WHEN year >= 2500 THEN 'Under 25'
                            WHEN year >= 2480 THEN '25-45'
                            WHEN year >= 2460 THEN '46-65'
                            WHEN year >= 2440 THEN '66-85'
                            ELSE 'Over 85'
                        END
                    ORDER BY 
                        CASE Age_Group
                            WHEN 'Under 25' THEN 1
                            WHEN '25-45' THEN 2
                            WHEN '46-65' THEN 3
                            WHEN '66-85' THEN 4
                            ELSE 5
                        END
                """, conn)
                df_age_groups.to_excel(writer, sheet_name='Age_Group_Analysis', index=False)
            
            conn.close()
            self.show_snack_bar(f"📊 Comprehensive voter analysis report exported successfully to: {file_path}", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error exporting voter analysis report: {ex}", ft.Colors.RED_600)

    # IMPORT IDENTITY CARD FUNCTIONS FOR TAB 12
    def on_identity_file_picked(self, e):
        """Handle identity file picker result"""
        if e.files:
            self.selected_identity_file = e.files[0].path
            self.identity_file_path_text.value = f"📁 {e.files[0].name}"
            self.identity_file_path_text.update()
            self.identity_import_status.value = f"File selected: {e.files[0].name}"
            self.identity_import_status.update()
        else:
            self.selected_identity_file = None
            self.identity_file_path_text.value = "ยังไม่ได้เลือกไฟล์"
            self.identity_file_path_text.update()
            self.identity_import_status.value = "พร้อมนำเข้าข้อมูลบัตรประชาชน..."
            self.identity_import_status.update()

    def preview_identity_file(self, e):
        """Preview and validate Excel/CSV file format"""
        if not hasattr(self, 'selected_identity_file') or not self.selected_identity_file:
            self.show_snack_bar("Please select an Excel or CSV file first", ft.Colors.ORANGE_600)
            return
        
        if not Path(self.selected_identity_file).exists():
            self.show_snack_bar("Selected file does not exist", ft.Colors.RED_600)
            return
        
        try:
            # Read file based on extension
            file_path = Path(self.selected_identity_file)
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.selected_identity_file)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.selected_identity_file)
            else:
                self.show_snack_bar("Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv)", ft.Colors.RED_600)
                return
            
            # Expected columns for election_c table
            expected_columns = [
                'order_no', 'identity_no', 'name_title', 'name', 'surname', 
                'no_address', 'mo_address', 'day_birth', 'month', 'year', 
                'sex', 'remark', 'birthday'
            ]
            
            # Validate columns
            missing_columns = []
            for col in expected_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            extra_columns = []
            for col in df.columns:
                if col not in expected_columns:
                    extra_columns.append(col)
            
            # Show validation results
            validation_message = f"📋 File validation results:\n"
            validation_message += f"• Total rows: {len(df)}\n"
            validation_message += f"• Total columns: {len(df.columns)}\n"
            
            if missing_columns:
                validation_message += f"❌ Missing columns: {', '.join(missing_columns)}\n"
            else:
                validation_message += f"✅ All required columns present\n"
            
            if extra_columns:
                validation_message += f"⚠️ Extra columns (will be ignored): {', '.join(extra_columns)}\n"
            
            # Clear preview table and populate with first 10 rows
            self.identity_preview_table.rows.clear()
            
            preview_data = df.head(10)
            for index, row in preview_data.iterrows():
                cells = []
                for col in expected_columns:
                    value = str(row.get(col, "N/A")) if col in df.columns else "N/A"
                    cells.append(ft.DataCell(ft.Text(value)))
                
                self.identity_preview_table.rows.append(ft.DataRow(cells=cells))
            
            self.identity_preview_table.update()
            
            # Update status
            self.identity_import_status.value = validation_message
            self.identity_import_status.update()
            
            if not missing_columns:
                self.show_snack_bar(f"✅ File validated successfully! {len(df)} rows ready to import", ft.Colors.GREEN_600)
            else:
                self.show_snack_bar(f"❌ Validation failed: Missing columns", ft.Colors.RED_600)
            
        except Exception as ex:
            error_msg = f"❌ Error reading file: {ex}"
            self.identity_import_status.value = error_msg
            self.identity_import_status.update()
            self.show_snack_bar(error_msg, ft.Colors.RED_600)

    def import_identity_data(self, e):
        """Import Excel/CSV data into election_c table"""
        if not hasattr(self, 'selected_identity_file') or not self.selected_identity_file:
            self.show_snack_bar("Please select and preview a file first", ft.Colors.ORANGE_600)
            return
        
        if not Path(self.selected_identity_file).exists():
            self.show_snack_bar("Selected file does not exist", ft.Colors.RED_600)
            return
        
        # Show progress
        self.identity_progress_bar.visible = True
        self.identity_progress_bar.update()
        self.identity_import_status.value = "Starting import..."
        self.identity_import_status.update()
        
        try:
            # Read file
            file_path = Path(self.selected_identity_file)
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(self.selected_identity_file)
            else:
                df = pd.read_csv(self.selected_identity_file)
            
            # Expected columns
            expected_columns = [
                'order_no', 'identity_no', 'name_title', 'name', 'surname', 
                'no_address', 'mo_address', 'day_birth', 'month', 'year', 
                'sex', 'remark', 'birthday'
            ]
            
            # Validate required columns exist
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                self.show_snack_bar(f"❌ Cannot import: Missing columns: {', '.join(missing_columns)}", ft.Colors.RED_600)
                self.identity_progress_bar.visible = False
                self.identity_progress_bar.update()
                return
            
            # Filter DataFrame to only include expected columns
            df_filtered = df[expected_columns].copy()
            
            # Connect to database and insert data
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            imported_count = 0
            error_count = 0
            errors = []
            
            self.identity_import_status.value = f"Importing {len(df_filtered)} records..."
            self.identity_import_status.update()
            
            for index, row in df_filtered.iterrows():
                try:
                    # Prepare values, converting NaN to None
                    values = []
                    for col in expected_columns:
                        val = row[col]
                        if pd.isna(val):
                            values.append(None)
                        else:
                            values.append(str(val))
                    
                    # Insert into election_c table
                    cursor.execute("""
                        INSERT INTO election_c (order_no, identity_no, name_title, name, surname, 
                                              no_address, mo_address, day_birth, month, year, 
                                              sex, remark, birthday)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, values)
                    
                    imported_count += 1
                    
                except Exception as row_error:
                    error_count += 1
                    errors.append(f"Row {index + 2}: {str(row_error)}")
                    if len(errors) < 10:  # Limit error collection
                        continue
            
            conn.commit()
            conn.close()
            
            # Update status
            success_message = f"✅ Import completed!\n"
            success_message += f"• Successfully imported: {imported_count} records\n"
            success_message += f"• Errors: {error_count} records\n"
            
            if errors:
                success_message += f"\nFirst few errors:\n"
                for error in errors[:5]:
                    success_message += f"• {error}\n"
            
            self.identity_import_status.value = success_message
            self.identity_import_status.update()
            
            if imported_count > 0:
                self.show_snack_bar(f"✅ Successfully imported {imported_count} identity records!", ft.Colors.GREEN_600)
                # รีเฟรชสถิติ current data display
                self.load_current_identity_data(None)
            else:
                self.show_snack_bar(f"❌ Import failed: No records imported", ft.Colors.RED_600)
            
        except Exception as ex:
            error_msg = f"❌ Import error: {ex}"
            self.identity_import_status.value = error_msg
            self.identity_import_status.update()
            self.show_snack_bar(error_msg, ft.Colors.RED_600)
        
        finally:
            self.identity_progress_bar.visible = False
            self.identity_progress_bar.update()

    def load_current_identity_data(self, e):
        """Load and display current data from election_c table"""
        try:
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get latest 20 records
            cursor.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                ORDER BY order_no DESC 
                LIMIT 20
            """)
            current_records = cursor.fetchall()
            conn.close()
            
            # Clear and populate current data table
            self.current_identity_table.rows.clear()
            
            for record in current_records:
                cells = []
                for value in record:
                    cells.append(ft.DataCell(ft.Text(str(value or ""))))
                
                self.current_identity_table.rows.append(ft.DataRow(cells=cells))
            
            self.current_identity_table.update()
            
            self.show_snack_bar(f"📋 Loaded {len(current_records)} current records from database", ft.Colors.BLUE_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading current data: {ex}", ft.Colors.RED_600)

    # SHOW BIRTH PLACE TAB FUNCTIONS
    def create_show_birth_place_tab(self):
        """Create Show Birth Place tab using SQL view to match identity numbers with birth places"""
        
        # Create show birth place table
        self.show_birth_place_table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Identity No", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Name", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("นามสกุล", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Birth Code", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("District", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Province", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("Details", weight=ft.FontWeight.BOLD)),
            ],
            rows=[],
            data_row_max_height=50,
            horizontal_lines=ft.BorderSide(1, ft.Colors.GREY_300),
            vertical_lines=ft.BorderSide(1, ft.Colors.GREY_300)
        )
        
        # Search field
        self.show_birth_place_search_field = ft.TextField(
            label="🔍 Search by Identity Number or Name",
            hint_text="Enter identity number, name, or surname",
            width=400,
            on_change=self.search_show_birth_place_records
        )
        
        # Details panel
        self.show_birth_place_details = ft.Column([
            ft.Text("👆 Click on any record above to see details...", 
                   size=16, color=ft.Colors.GREY_600)
        ])
        
        return ft.Column([
            ft.Card(elevation=4, content=ft.Container(content=ft.Row([
                ft.Icon(ft.Icons.MAP, size=30, color=ft.Colors.GREEN_600), 
                ft.Text("🌍 แสดงสถานที่เกิด", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_800), 
                ft.Icon(ft.Icons.LOCATION_ON, size=30, color=ft.Colors.GREEN_600),
            ], alignment=ft.MainAxisAlignment.CENTER), padding=15, bgcolor=ft.Colors.GREEN_100)),
            
            ft.Card(elevation=4, content=ft.Container(content=ft.Column([
                                        ft.Text("🔍 ค้นหาเลขประจำตัวพร้อมสถานที่เกิด", size=16, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                ft.Row([
                    self.show_birth_place_search_field
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([
                    ft.ElevatedButton("✅ แสดงเฉพาะข้อมูลที่ตรงกัน", 
                                    bgcolor=ft.Colors.BLUE_600, 
                                    color=ft.Colors.WHITE, 
                                    on_click=self.load_matched_show_birth_place_records, 
                                    icon=ft.Icons.CHECK_CIRCLE),
                                            ft.Text("👈 คลิกเพื่อดูเฉพาะข้อมูลที่มีสถานที่เกิด", 
                           size=12, color=ft.Colors.BLUE_600)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([
                    ft.ElevatedButton("📊 แสดงสถิติ", 
                                    bgcolor=ft.Colors.BLUE_600, 
                                    color=ft.Colors.WHITE, 
                                    on_click=self.show_birth_place_stats, 
                                    icon=ft.Icons.ANALYTICS),
                    ft.ElevatedButton("📄 ส่งออกข้อมูล", 
                                    bgcolor=ft.Colors.ORANGE_600, 
                                    color=ft.Colors.WHITE, 
                                    on_click=self.export_show_birth_place_data, 
                                    icon=ft.Icons.DOWNLOAD)
                ], alignment=ft.MainAxisAlignment.SPACE_AROUND)
            ], spacing=15), padding=20, bgcolor=ft.Colors.BLUE_50)),
            
            ft.Card(elevation=4, content=ft.Container(content=ft.Column([
                                        ft.Text("📋 เลขประจำตัวพร้อมข้อมูลสถานที่เกิด", size=16, weight=ft.FontWeight.BOLD),
                ft.Text("👆 เลื่อนแนวนอนและแนวตั้งเพื่อดูข้อมูลทั้งหมด | คลิกบนแถวเพื่อดูรายละเอียด", 
                       size=12, color=ft.Colors.GREEN_600),
                ft.Divider(),
                ft.Container(
                    content=ft.Row([
                        ft.Column([self.show_birth_place_table], scroll=ft.ScrollMode.ALWAYS, expand=True)
                    ], scroll=ft.ScrollMode.ALWAYS, expand=True),
                    height=400,
                    bgcolor=ft.Colors.BLUE_50,
                    border=ft.border.all(1, ft.Colors.BLUE_300),
                    border_radius=8,
                    padding=10
                )
            ]), padding=15, bgcolor=ft.Colors.BLUE_50)),
            
            ft.Card(elevation=4, content=ft.Container(content=ft.Column([
                ft.Text("📝 รายละเอียดข้อมูล", size=16, weight=ft.FontWeight.BOLD),
                ft.Divider(),
                self.show_birth_place_details
            ]), padding=20, bgcolor=ft.Colors.BLUE_100))
        ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)
    
    def load_all_show_birth_place_records(self, e):
        """Load all records from election_c table and match with birth places"""
        try:
            # Use election_c table (same as ID Show tab) and match with id_birthplace
            conn1 = sqlite3.connect('demo_voters.db')
            conn2 = sqlite3.connect('demo_geo_voters.db')
            
            cursor1 = conn1.cursor()
            cursor2 = conn2.cursor()
            
            # Get ALL data from election_c (same as ID Show tab)
            cursor1.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no IS NOT NULL AND identity_no != ''
                ORDER BY order_no
                LIMIT 100
            """)
            
            election_records = cursor1.fetchall()
            conn1.close()
            
            # Clear table
            self.show_birth_place_table.rows.clear()
            
            for election_record in election_records:
                order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday = election_record
                
                # Extract birth code from identity number (e.g., "3-6405-00108-25-0" -> "6405")
                birth_code = None
                if '-' in identity_no:
                    parts = identity_no.split('-')
                    if len(parts) >= 2:
                        birth_code = parts[1]
                
                # Get birth place information
                district = "N/A"
                province = "N/A"
                
                if birth_code:
                    # Try matching with integer first (as Code column is INTEGER)
                    try:
                        birth_code_int = int(birth_code)
                        cursor2.execute("""
                            SELECT District, Province 
                            FROM id_birthplace 
                            WHERE Code = ?
                        """, (birth_code_int,))
                        
                        birth_place_result = cursor2.fetchone()
                        if birth_place_result:
                            district, province = birth_place_result
                    except ValueError:
                        # Skip invalid birth codes
                        pass
                
                # Create Details button
                details_button = ft.ElevatedButton(
                    "👁️",
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.INDIGO_600,
                    width=50,
                    height=30,
                    on_click=lambda e, identity=identity_no: self.on_show_birth_place_record_select(identity)
                )
                
                # Add row to table
                self.show_birth_place_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(identity_no or ""))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(birth_code or "N/A"))),
                            ft.DataCell(ft.Text(str(district or "N/A"))),
                            ft.DataCell(ft.Text(str(province or "N/A"))),
                            ft.DataCell(details_button),
                        ]
                    )
                )
            
            conn2.close()
            
            self.show_birth_place_table.update()
            self.show_snack_bar(f"📋 Loaded {len(election_records)} records with birth place information", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading Show Birth Place records: {ex}", ft.Colors.RED_600)
    
    def load_matched_show_birth_place_records(self, e):
        """Load only records that have matching birth place data"""
        try:
            # Use SQL view with JOIN between election_c and id_birthplace tables
            conn1 = sqlite3.connect('demo_voters.db')
            conn2 = sqlite3.connect('demo_geo_voters.db')
            
            cursor1 = conn1.cursor()
            cursor2 = conn2.cursor()
            
            # Get data from election_c (same as ID Show tab)
            cursor1.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no IS NOT NULL AND identity_no != ''
                ORDER BY order_no
                LIMIT 200
            """)
            
            election_records = cursor1.fetchall()
            conn1.close()
            
            # Clear table
            self.show_birth_place_table.rows.clear()
            
            matched_count = 0
            for election_record in election_records:
                order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday = election_record
                
                # Extract birth code from identity number
                birth_code = None
                if '-' in identity_no:
                    parts = identity_no.split('-')
                    if len(parts) >= 2:
                        birth_code = parts[1]
                
                # Get birth place information - only add if match found
                district = None
                province = None
                
                if birth_code:
                    try:
                        birth_code_int = int(birth_code)
                        cursor2.execute("""
                            SELECT District, Province 
                            FROM id_birthplace 
                            WHERE Code = ?
                        """, (birth_code_int,))
                        
                        birth_place_result = cursor2.fetchone()
                        if birth_place_result:
                            district, province = birth_place_result
                            
                            # Create Details button
                            details_button = ft.ElevatedButton(
                                "👁️",
                                color=ft.Colors.WHITE,
                                bgcolor=ft.Colors.INDIGO_600,
                                width=50,
                                height=30,
                                on_click=lambda e, identity=identity_no: self.on_show_birth_place_record_select(identity)
                            )
                            
                            # Only add row if we found a match
                            self.show_birth_place_table.rows.append(
                                ft.DataRow(
                                    cells=[
                                        ft.DataCell(ft.Text(str(identity_no or ""))),
                                        ft.DataCell(ft.Text(str(name or ""))),
                                        ft.DataCell(ft.Text(str(surname or ""))),
                                        ft.DataCell(ft.Text(str(birth_code or "N/A"))),
                                        ft.DataCell(ft.Text(str(district or "N/A"))),
                                        ft.DataCell(ft.Text(str(province or "N/A"))),
                                        ft.DataCell(details_button),
                                    ]
                                )
                            )
                            matched_count += 1
                    except ValueError:
                        # Skip invalid birth codes
                        pass
            
            conn2.close()
            
            self.show_birth_place_table.update()
            self.show_snack_bar(f"✅ Loaded {matched_count} records with matching birth place information", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading matched Show Birth Place records: {ex}", ft.Colors.RED_600)
    
    def search_show_birth_place_records(self, e):
        """Search Show Birth Place records based on search term"""
        search_term = self.show_birth_place_search_field.value.strip()
        if not search_term:
            return
        
        try:
            # Use SQL view with JOIN between election_c and id_birthplace tables
            conn1 = sqlite3.connect('demo_voters.db')
            conn2 = sqlite3.connect('demo_geo_voters.db')
            
            cursor1 = conn1.cursor()
            cursor2 = conn2.cursor()
            
            # Search data from election_c (same as ID Show tab)
            cursor1.execute("""
                SELECT order_no, identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE (identity_no LIKE ? OR name LIKE ? OR surname LIKE ?)
                AND identity_no IS NOT NULL AND identity_no != ''
                ORDER BY order_no
                LIMIT 50
            """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
            
            election_records = cursor1.fetchall()
            conn1.close()
            
            # Clear table
            self.show_birth_place_table.rows.clear()
            
            for election_record in election_records:
                order_no, identity_no, name_title, name, surname, no_address, mo_address, day_birth, month, year, sex, remark, birthday = election_record
                
                # Extract birth code from identity number
                birth_code = None
                if '-' in identity_no:
                    parts = identity_no.split('-')
                    if len(parts) >= 2:
                        birth_code = parts[1]
                
                # Get birth place information
                district = "N/A"
                province = "N/A"
                
                if birth_code:
                    # Try matching with integer first (as Code column is INTEGER)
                    try:
                        birth_code_int = int(birth_code)
                        cursor2.execute("""
                            SELECT District, Province 
                            FROM id_birthplace 
                            WHERE Code = ?
                        """, (birth_code_int,))
                        
                        birth_place_result = cursor2.fetchone()
                        if birth_place_result:
                            district, province = birth_place_result
                    except ValueError:
                        # Skip invalid birth codes
                        pass
                
                # Create Details button
                details_button = ft.ElevatedButton(
                    "👁️",
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.INDIGO_600,
                    width=50,
                    height=30,
                    on_click=lambda e, identity=identity_no: self.on_show_birth_place_record_select(identity)
                )
                
                # Add row to table
                self.show_birth_place_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(identity_no or ""))),
                            ft.DataCell(ft.Text(str(name or ""))),
                            ft.DataCell(ft.Text(str(surname or ""))),
                            ft.DataCell(ft.Text(str(birth_code or "N/A"))),
                            ft.DataCell(ft.Text(str(district or "N/A"))),
                            ft.DataCell(ft.Text(str(province or "N/A"))),
                            ft.DataCell(details_button),
                        ]
                    )
                )
            
            conn2.close()
            
            self.show_birth_place_table.update()
            self.show_snack_bar(f"🔍 Found {len(election_records)} matching records for '{search_term}'", ft.Colors.BLUE_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error searching records: {ex}", ft.Colors.RED_600)
    
    def on_show_birth_place_record_select(self, identity_no):
        """Handle record selection in Show Birth Place tab"""
        try:
            # Get full record details from both databases
            conn1 = sqlite3.connect('demo_voters.db')
            conn2 = sqlite3.connect('demo_geo_voters.db')
            
            cursor1 = conn1.cursor()
            cursor2 = conn2.cursor()
            
            # Get identity record details
            cursor1.execute("""
                SELECT identity_no, name_title, name, surname, no_address, mo_address, 
                       day_birth, month, year, sex, remark, birthday
                FROM election_c 
                WHERE identity_no = ?
            """, (identity_no,))
            
            identity_record = cursor1.fetchone()
            conn1.close()
            
            if identity_record:
                identity_no, title, name, surname, address_no, mo_address, day_birth, month, year, sex, remark, birthday = identity_record
                
                # Extract birth code
                birth_code = None
                if '-' in identity_no:
                    parts = identity_no.split('-')
                    if len(parts) >= 2:
                        birth_code = parts[1]
                
                # Get birth place details
                district = "N/A"
                province = "N/A"
                
                if birth_code:
                    # Convert to integer for matching
                    try:
                        birth_code_int = int(birth_code)
                        cursor2.execute("""
                            SELECT Code, District, Province, field4
                            FROM id_birthplace 
                            WHERE Code = ?
                        """, (birth_code_int,))
                    except ValueError:
                        # Try as string if integer conversion fails
                        cursor2.execute("""
                            SELECT Code, District, Province, field4
                            FROM id_birthplace 
                            WHERE Code = ?
                        """, (birth_code,))
                    
                    birth_place_record = cursor2.fetchone()
                    if birth_place_record:
                        code, district, province, field4 = birth_place_record
                
                conn2.close()
                
                # Format details for modal dialog
                details_text = f"🌍 Identity & Birth Place Details: {identity_no}\n\n"
                details_text += "=" * 60 + "\n\n"
                
                details_text += "📋 Identity Information:\n"
                details_text += f"• Identity No: {identity_no}\n"
                details_text += f"• Title: {title or 'N/A'}\n"
                details_text += f"• Name: {name or 'N/A'}\n"
                details_text += f"• Surname: {surname or 'N/A'}\n"
                details_text += f"• Gender: {sex or 'N/A'}\n\n"
                
                details_text += "🏠 Address Information:\n"
                details_text += f"• Address No: {address_no or 'N/A'}\n"
                details_text += f"• หมู่ที่: {mo_address or 'N/A'}\n"
                details_text += f"• Remark: {remark or 'N/A'}\n\n"
                
                details_text += "🌍 Birth Place Information:\n"
                details_text += f"• Birth Code: {birth_code or 'N/A'}\n"
                details_text += f"• District: {district}\n"
                details_text += f"• Province: {province}\n"
                details_text += f"• Birth Date: {day_birth or 'N/A'}/{month or 'N/A'}/{year or 'N/A'}\n"
                
                # Show details in modal dialog instead of panel
                self.show_desktop_modal_dialog(f"🌍 รายละเอียดสถานที่เกิด: {identity_no}", details_text, 650, 700)
            else:
                self.show_snack_bar("❌ Record not found", ft.Colors.RED_600)
                
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading record details: {ex}", ft.Colors.RED_600)
    
    def show_birth_place_stats(self, e):
        """Show statistics for Show Birth Place data in a modal dialog"""
        try:
            conn1 = sqlite3.connect('demo_voters.db')
            conn2 = sqlite3.connect('demo_geo_voters.db')
            
            cursor1 = conn1.cursor()
            cursor2 = conn2.cursor()
            
            # Get various statistics
            cursor1.execute("SELECT COUNT(id) FROM election_c WHERE identity_no IS NOT NULL AND identity_no != ''")
            total_identities = cursor1.fetchone()[0]
            
            cursor1.execute("SELECT COUNT(DISTINCT SUBSTR(identity_no, INSTR(identity_no, '-') + 1, INSTR(SUBSTR(identity_no, INSTR(identity_no, '-') + 1), '-') - 1)) FROM election_c WHERE identity_no IS NOT NULL AND identity_no != '' AND identity_no LIKE '%-%'")
            unique_birth_codes = cursor1.fetchone()[0]
            
            cursor2.execute("SELECT COUNT(*) FROM id_birthplace")
            total_birth_places = cursor2.fetchone()[0]
            
            cursor2.execute("SELECT COUNT(DISTINCT Province) FROM id_birthplace WHERE Province IS NOT NULL")
            unique_provinces = cursor2.fetchone()[0]
            
            # Get matched records count
            cursor1.execute("SELECT COUNT(*) FROM election_c WHERE identity_no IS NOT NULL AND identity_no != '' AND identity_no LIKE '%-%'")
            matched_records = cursor1.fetchone()[0]
            
            # Get unmatched records count
            cursor1.execute("SELECT COUNT(*) FROM election_c WHERE identity_no IS NOT NULL AND identity_no != '' AND (identity_no NOT LIKE '%-%' OR identity_no = '')")
            unmatched_records = cursor1.fetchone()[0]
            
            conn1.close()
            conn2.close()
            
            stats_content = f"""📊 สถิติข้อมูลสถานที่เกิด:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 จำนวนบัตรประชาชนทั้งหมด: {total_identities:,} รายการ
🏷️ รหัสสถานที่เกิดที่ไม่ซ้ำ: {unique_birth_codes:,} รหัส
🌍 จำนวนข้อมูลสถานที่เกิด: {total_birth_places:,} รายการ
🏛️ จำนวนจังหวัดที่ไม่ซ้ำ: {unique_provinces:,} จังหวัด
✅ บัตรที่มีรหัสสถานที่เกิด: {matched_records:,} รายการ
❌ บัตรที่ไม่มีรหัสสถานที่เกิด: {unmatched_records:,} รายการ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 หมายเหตุ: ข้อมูลแสดงการจับคู่ระหว่างบัตรประชาชนกับข้อมูลสถานที่เกิด"""
            
            # Show statistics in modal dialog
            self.show_desktop_modal_dialog("📊 สถิติสถานที่เกิด", stats_content, 600, 500)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error calculating statistics: {ex}", ft.Colors.RED_600)
    
    def export_show_birth_place_data(self, e):
        """Export Show Birth Place data to Excel"""
        try:
            # Create combined data using SQL view
            conn1 = sqlite3.connect('demo_voters.db')
            conn2 = sqlite3.connect('demo_geo_voters.db')
            
            # Get data from election_c table (same as ID Show tab)
            election_df = pd.read_sql_query("""
                SELECT order_no as 'Order No',
                       identity_no as 'Identity Number', 
                       name_title as 'Title',
                       name as 'Name', 
                       surname as 'Surname',
                       no_address as 'Address No',
                       mo_address as 'หมู่ที่',
                       day_birth as 'Day Birth',
                       month as 'Month',
                       year as 'Year',
                       sex as 'Gender',
                       remark as 'Remark',
                       birthday as 'Birthday'
                FROM election_c 
                WHERE identity_no IS NOT NULL AND identity_no != ''
                ORDER BY order_no
            """, conn1)
            
            conn1.close()
            
            # Extract birth codes and get birth place data
            birth_place_data = []
            cursor2 = conn2.cursor()
            
            for index, row in election_df.iterrows():
                identity_no = row['Identity Number']
                birth_code = None
                
                if '-' in identity_no:
                    parts = identity_no.split('-')
                    if len(parts) >= 2:
                        birth_code = parts[1]
                
                district = "N/A"
                province = "N/A"
                
                if birth_code:
                    try:
                        birth_code_int = int(birth_code)
                        cursor2.execute("SELECT District, Province FROM id_birthplace WHERE Code = ?", (birth_code_int,))
                    except ValueError:
                        cursor2.execute("SELECT District, Province FROM id_birthplace WHERE Code = ?", (birth_code,))
                    result = cursor2.fetchone()
                    if result:
                        district, province = result
                
                birth_place_data.append({
                    'Birth Code': birth_code or 'N/A',
                    'District': district,
                    'Province': province
                })
            
            conn2.close()
            
            # Combine data
            birth_place_df = pd.DataFrame(birth_place_data)
            combined_df = pd.concat([election_df, birth_place_df], axis=1)
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"show_birth_place_report_{timestamp}.xlsx"
            
            # Export to Excel
            combined_df.to_excel(filename, index=False)
            
            self.show_snack_bar(f"📄 Data exported successfully to {filename}", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error exporting data: {ex}", ft.Colors.RED_600)

    # ID BIRTH PLACE TAB FUNCTIONS
    def load_all_id_birthplace_records(self, e):
        """Load all records from id_birthplace table in demo_geo_voters.db"""
        try:
            conn = sqlite3.connect('demo_geo_voters.db')
            cursor = conn.cursor()
            
            # Get all records from id_birthplace table
            cursor.execute("""
                SELECT Code, District, Province, field4
                FROM id_birthplace 
                ORDER BY Code
                LIMIT 100
            """)
            
            records = cursor.fetchall()
            conn.close()
            
            # Clear and populate table
            self.id_birthplace_table.rows.clear()
            
            for record in records:
                code, district, province, field4 = record
                
                # Create Details button
                details_button = ft.ElevatedButton(
                    "👁️",
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.PURPLE_600,
                    width=50,
                    height=30,
                    on_click=lambda e, code=code: self.on_id_birthplace_record_select(code)
                )
                
                self.id_birthplace_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(code or ""))),
                            ft.DataCell(ft.Text(str(district or "N/A"))),
                            ft.DataCell(ft.Text(str(province or "N/A"))),
                            ft.DataCell(ft.Text(str(field4 or "N/A"))),
                            ft.DataCell(details_button),
                        ]
                    )
                )
            
            self.id_birthplace_table.update()
            self.show_snack_bar(f"📋 Loaded {len(records)} ID Birth Place records", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading ID Birth Place records: {ex}", ft.Colors.RED_600)
    
    def search_id_birthplace_records(self, e):
        """Search ID Birth Place records based on search term"""
        search_term = self.id_birthplace_search_field.value.strip()
        if not search_term:
            return
        
        try:
            conn = sqlite3.connect('demo_geo_voters.db')
            cursor = conn.cursor()
            
            # Search records
            cursor.execute("""
                SELECT Code, District, Province, field4
                FROM id_birthplace 
                WHERE Code LIKE ? OR District LIKE ? OR Province LIKE ? OR field4 LIKE ?
                ORDER BY Code
                LIMIT 50
            """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
            
            records = cursor.fetchall()
            conn.close()
            
            # Clear and populate table
            self.id_birthplace_table.rows.clear()
            
            for record in records:
                code, district, province, field4 = record
                
                # Create Details button
                details_button = ft.ElevatedButton(
                    "👁️",
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.PURPLE_600,
                    width=50,
                    height=30,
                    on_click=lambda e, code=code: self.on_id_birthplace_record_select(code)
                )
                
                self.id_birthplace_table.rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(code or ""))),
                            ft.DataCell(ft.Text(str(district or "N/A"))),
                            ft.DataCell(ft.Text(str(province or "N/A"))),
                            ft.DataCell(ft.Text(str(field4 or "N/A"))),
                            ft.DataCell(details_button),
                        ]
                    )
                )
            
            self.id_birthplace_table.update()
            self.show_snack_bar(f"🔍 Found {len(records)} matching records for '{search_term}'", ft.Colors.BLUE_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error searching records: {ex}", ft.Colors.RED_600)
    
    def on_id_birthplace_record_select(self, code):
        """Handle record selection in ID Birth Place tab"""
        try:
            conn = sqlite3.connect('demo_geo_voters.db')
            cursor = conn.cursor()
            
            # Get full record details
            cursor.execute("""
                SELECT Code, District, Province, field4
                FROM id_birthplace 
                WHERE Code = ?
            """, (code,))
            
            record = cursor.fetchone()
            conn.close()
            
            if record:
                code, district, province, field4 = record
                
                # Format details for modal dialog
                details_text = f"🌍 Birth Place Code Details: {code}\n\n"
                details_text += "=" * 50 + "\n\n"
                
                details_text += "🏷️ Birth Place Code Information:\n"
                details_text += f"• Code: {code}\n"
                details_text += f"• Type: {'Province' if str(code).endswith('00') else 'District/Area'}\n\n"
                
                details_text += "📍 Location Information:\n"
                details_text += f"• District: {district or 'N/A'}\n"
                details_text += f"• Province: {province or 'N/A'}\n"
                details_text += f"• Additional Info: {field4 or 'Not specified'}\n\n"
                
                details_text += "📊 Statistics:\n"
                details_text += "• Click 'Show Statistics' button to see data for all locations\n"
                
                # Show details in modal dialog instead of panel
                self.show_desktop_modal_dialog(f"🌍 รหัสสถานที่เกิด: {code}", details_text, 550, 600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading record details: {ex}", ft.Colors.RED_600)
    
    def show_id_birthplace_stats(self, e):
        """Show statistics for ID Birth Place data in a modal dialog"""
        try:
            conn = sqlite3.connect('demo_geo_voters.db')
            cursor = conn.cursor()
            
            # Get various statistics
            cursor.execute("SELECT COUNT(*) FROM id_birthplace")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT Province) FROM id_birthplace WHERE Province IS NOT NULL AND Province != ''")
            unique_provinces = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT District) FROM id_birthplace WHERE District IS NOT NULL AND District != ''")
            unique_districts = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM id_birthplace WHERE Code LIKE '%00'")
            province_codes = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM id_birthplace WHERE Code NOT LIKE '%00'")
            district_codes = cursor.fetchone()[0]
            
            conn.close()
            
            stats_content = f"""📊 สถิติข้อมูลสถานที่เกิด:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 จำนวนข้อมูลสถานที่เกิดทั้งหมด: {total_records:,} รายการ
🏛️ จำนวนจังหวัดที่ไม่ซ้ำ: {unique_provinces:,} จังหวัด
🏘️ จำนวนอำเภอที่ไม่ซ้ำ: {unique_districts:,} อำเภอ
🏷️ รหัสจังหวัด (ลงท้ายด้วย 00): {province_codes:,} รหัส
🏷️ รหัสอำเภอ/พื้นที่: {district_codes:,} รหัส
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 หมายเหตุ: ข้อมูลแสดงสถิติของตาราง id_birthplace"""
            
            # Show statistics in modal dialog
            self.show_desktop_modal_dialog("📊 สถิติข้อมูลสถานที่เกิด", stats_content, 600, 500)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading statistics: {ex}", ft.Colors.RED_600)
    
    def export_id_birthplace_data(self, e):
        """Export ID Birth Place data to Excel"""
        try:
            conn = sqlite3.connect('demo_geo_voters.db')
            
            # Query all data
            df = pd.read_sql_query("""
                SELECT Code as 'Birth Place Code', 
                       District as 'District Name',
                       Province as 'Province Name', 
                       field4 as 'Additional Information'
                FROM id_birthplace
                ORDER BY Code
            """, conn)
            
            conn.close()
            
            # Export to Excel
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"id_birthplace_report_{timestamp}.xlsx"
            
            df.to_excel(filename, index=False, engine='openpyxl')
            
            self.show_snack_bar(f"📄 ID Birth Place data exported to: {filename}", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error exporting data: {ex}", ft.Colors.RED_600)

    def load_initial_data(self):
        """Load initial data when app starts"""
        # Load statistics on startup
        self.update_stats(None)
        # Load Show Birth Place data automatically on startup
        self.load_all_show_birth_place_records(None)
        # Load birthplace data automatically on startup
        self.load_birthplace_data(None)

    def create_support_level_dropdown(self, current_value, surname):
        """Create a dropdown for support level selection in datagridview"""
        return ft.Dropdown(
            value=current_value or "unknown",
            options=[
                ft.dropdown.Option("strong_support", "Strong Support"),
                ft.dropdown.Option("moderate_support", "Moderate Support"),
                ft.dropdown.Option("neutral", "Neutral"),
                ft.dropdown.Option("opposition", "Opposition"),
                ft.dropdown.Option("unknown", "Unknown")
            ],
            width=200,
            border_color=ft.Colors.GREEN_200,
            focused_border_color=ft.Colors.GREEN_500,
            on_change=lambda e, s=surname: self.on_support_level_change(e, s)
        )
    

    
    def on_support_level_change(self, e, surname):
        """Handle support level change in datagridview dropdown"""
        try:
            new_support_level = e.control.value
            if new_support_level:
                # Update the database
                self.db.update_family_voter_support(surname, new_support_level)
                
                # Update the dropdown value to reflect the change
                e.control.value = new_support_level
                e.control.update()
                
                # รีเฟรชสถิติ the database panel to show updated data
                self.refresh_family_voter_database()
                
                self.show_snack_bar(f"✅ Updated support level for '{surname}' to '{new_support_level}'", ft.Colors.GREEN_600)
        except Exception as ex:
            self.show_snack_bar(f"❌ Error updating support level: {ex}", ft.Colors.RED_600)
    


    def refresh_family_voter_database(self):
        """รีเฟรชสถิติ the family voter database panel with data from surname table"""
        self.load_surname_data()

    def load_surname_data(self):
        """Load basic family data from election_c table with notes from surname table"""
        print("🔧 DEBUG: load_surname_data called")
        try:
            # Connect to database
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get basic family data with notes from surname table
            cursor.execute("""
                SELECT e.surname, COUNT(DISTINCT e.identity_no) as total_voters,
                       COUNT(DISTINCT e.no_address || '|' || e.mo_address) as unique_houses,
                       COALESCE(s.note1, '') as note1,
                       COALESCE(s.note2, '') as note2,
                       COALESCE(s.note3, '') as note3
                FROM election_c e
                LEFT JOIN surname s ON e.surname = s.surname
                WHERE e.surname IS NOT NULL AND e.surname != '' AND e.surname != 'N/A'
                  AND e.identity_no IS NOT NULL AND e.identity_no != '' AND e.identity_no != 'N/A'
                GROUP BY e.surname, s.note1, s.note2, s.note3
                HAVING COUNT(DISTINCT e.identity_no) >= 2
                ORDER BY total_voters DESC, e.surname ASC
                LIMIT 15
            """)
            
            family_data = cursor.fetchall()
            conn.close()
            
            print(f"🔧 DEBUG: Found {len(family_data)} family records")
            
            # Clear and populate the table
            if hasattr(self, 'family_voter_table'):
                self.family_voter_table.rows.clear()
                
                for surname, total_voters, houses, note1, note2, note3 in family_data:
                    details_button = ft.ElevatedButton(
                        "👁️",
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.BLUE_600,
                        width=50,
                        height=30,
                        on_click=lambda e, s=surname, tv=total_voters: self.on_family_select(s, tv)
                    )
                    
                    self.family_voter_table.rows.append(
                        ft.DataRow(cells=[
                            ft.DataCell(ft.Text(str(surname))),
                            ft.DataCell(ft.Text(str(total_voters))),
                            ft.DataCell(ft.Text(str(houses))),
                            ft.DataCell(ft.Text(str(note1))),  # note1 from surname table
                            ft.DataCell(ft.Text(str(note2))),  # note2 from surname table
                            ft.DataCell(ft.Text(str(note3))),  # note3 from surname table
                            ft.DataCell(details_button)
                        ])
                    )
                
                self.family_voter_table.update()
                print(f"🔧 DEBUG: Updated table with {len(family_data)} rows")
            else:
                print("🔧 DEBUG: family_voter_table not found")
            
        except Exception as ex:
            print(f"🔧 DEBUG: Error in load_surname_data: {ex}")

    def show_family_details_in_panel(self, surname, total_voters, houses, note1, note2, note3):
        """Show family details in the family detail panel when row is clicked"""
        print(f"🔧 DEBUG: Family details panel called for surname: {surname}")
        try:
            # Create simple detailed information text first
            details_text = f"🏷️ ชื่อสกุล: {surname}\n👥 จำนวนผู้มีสิทธิ์: {total_voters} คน\n🏠 จำนวนบ้าน: {houses} หลัง\n\n📝 หมายเหตุ:\n• หมายเหตุ 1: {note1 or 'ไม่มี'}\n• หมายเหตุ 2: {note2 or 'ไม่มี'}\n• หมายเหตุ 3: {note3 or 'ไม่มี'}"
            
            print(f"🔧 DEBUG: Created basic details text: {details_text[:100]}...")
            
            # Get additional details from election_c table for this surname
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Get voter details for this surname
            cursor.execute("""
                SELECT name, surname, sex, no_address, mo_address
                FROM election_c 
                WHERE surname = ? AND identity_no IS NOT NULL AND identity_no != '' AND identity_no != 'N/A'
                ORDER BY name ASC
                LIMIT 5
            """, (surname,))
            
            voter_details = cursor.fetchall()
            print(f"🔧 DEBUG: Found {len(voter_details)} voter details")
            
            conn.close()
            
            # Add voter details
            details_text += f"\n\n👤 ตัวอย่างผู้มีสิทธิ์ ({len(voter_details)} คน):\n"
            
            for i, (name, surname_db, sex, no_addr, mo_addr) in enumerate(voter_details, 1):
                sex_display = "ชาย" if sex == "M" else "หญิง" if sex == "F" else sex or "ไม่ระบุ"
                details_text += f"{i}. {name or ''} {surname_db or ''} ({sex_display}) - {no_addr or ''} หมู่ {mo_addr or ''}\n"
            
            if len(voter_details) == 0:
                details_text += "ไม่พบข้อมูลผู้มีสิทธิ์\n"
            
            # Update the family details panel
            print(f"🔧 DEBUG: Final details text length: {len(details_text)}")
            self.family_details.value = details_text
            self.family_details.update()
            print(f"🔧 DEBUG: Family details panel updated successfully")
            
        except Exception as ex:
            print(f"🔧 DEBUG: Error in family details: {ex}")
            error_text = f"❌ เกิดข้อผิดพลาด: {str(ex)}"
            self.family_details.value = error_text
            self.family_details.update()
            print(f"🔧 DEBUG: Error text set: {error_text}")

    def load_joined_family_data(self):
        """Load joined data from family_voter and surname tables (deprecated - now uses surname only)"""
        # Redirect to surname-only data loading
        self.load_surname_data()

    def load_family_voter_database(self, e):
        """Load data from family_voter table and display in the database panel"""
        try:
            # Connect directly to the same database file used by the main app
            conn = sqlite3.connect('demo_voters.db')
            cursor = conn.cursor()
            
            # Check if family_voter table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='family_voter'")
            if not cursor.fetchone():
                # Create the table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE family_voter (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        surname TEXT NOT NULL,
                        total_voters INTEGER DEFAULT 0,
                        houses_count INTEGER DEFAULT 0,
                        support_level TEXT DEFAULT 'unknown',
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
            
            # Load data from surname table only
            conn.close()
            self.load_surname_data()
            self.show_snack_bar("📋 Loaded joined family and surname data", ft.Colors.GREEN_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error loading family voter database: {ex}", ft.Colors.RED_600)
    
    def show_family_voter_stats(self, e):
        """Show statistics from family_voter table"""
        try:
            # Get all data from family_voter table
            family_voter_data = self.db.get_family_voter_data(1)
            
            if not family_voter_data:
                self.show_snack_bar("📊 No data found in family_voter table", ft.Colors.BLUE_600)
                return
            
            # Calculate statistics
            total_families = len(family_voter_data)
            total_voters = sum(family['total_voters'] for family in family_voter_data)
            total_houses = sum(family['houses_count'] for family in family_voter_data)
            
            # Support level statistics
            support_levels = {}
            for family in family_voter_data:
                level = family['support_level'] or 'unknown'
                support_levels[level] = support_levels.get(level, 0) + 1
            
            # Create statistics text
            stats_text = f"📊 Family Voter Database Statistics\n"
            stats_text += f"─" * 50 + "\n"
            stats_text += f"👨‍👩‍👧‍👦 Total Families: {total_families}\n"
            stats_text += f"👥 Total Voters: {total_voters}\n"
            stats_text += f"🏠 Total Houses: {total_houses}\n"
            stats_text += f"📈 Average Voters per Family: {total_voters/total_families:.1f}\n"
            stats_text += f"🏘️ Average Houses per Family: {total_houses/total_families:.1f}\n\n"
            
            stats_text += f"🎯 Support Level Distribution:\n"
            for level, count in support_levels.items():
                percentage = (count / total_families) * 100
                stats_text += f"  • {level}: {count} families ({percentage:.1f}%)\n"
            
            # Show statistics in a snack bar or update family details
            self.family_details.value = stats_text
            self.family_details.update()
            self.show_snack_bar(f"📊 Statistics calculated for {total_families} families", ft.Colors.PURPLE_600)
            
        except Exception as ex:
            self.show_snack_bar(f"❌ Error calculating statistics: {ex}", ft.Colors.RED_600)

def main(page: ft.Page):
    """Main entry point"""
    page.window_visible = True
    page.bgcolor = ft.Colors.BLUE_50
    page.update()
    app = FinalCompleteSidebarApp(page)

if __name__ == "__main__":
    print("🚀 Starting Final Complete Sidebar Election System...")
    print("=" * 60)
    print("🎯 Complete Functionality with Left Sidebar:")
    print("   • ALL original features preserved")
    print("   • NO DIALOGS - everything direct")
    print("   • Full database integration")
    print("   • Identity tab with complete functionality")
    print("=" * 60)
    
    ft.app(target=main) 