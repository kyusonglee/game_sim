import pygame
import random
import sys
import time

# --- CONFIG ---
SCREEN_W, SCREEN_H = 800, 600
ROOMS = ['Living Room', 'Bedroom', 'Kitchen', 'Bathroom']

# Expanded object system with more variety
OBJECT_TYPES = ['cup', 'book', 'ball', 'phone', 'keys', 'bottle', 'pen', 'glasses', 'wallet', 'watch']
OBJECT_COLORS = ['red', 'blue', 'yellow', 'green', 'purple', 'orange', 'pink', 'brown', 'black', 'white']
COLORS = {
    'red': (255,0,0), 
    'blue': (0,0,255), 
    'yellow': (255,255,0), 
    'gray': (150,150,150), 
    'green': (0,200,0),
    'purple': (128,0,128),
    'orange': (255,165,0),
    'pink': (255,192,203),
    'brown': (139,69,19),
    'black': (50,50,50),
    'white': (255,255,255),
    'wall': (80, 80, 80),
    'floor': (200, 180, 140),
    'door': (139, 69, 19)
}
FONT_SIZE = 20

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
font = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()

# --- DATA STRUCTURES ---
class Room:
    def __init__(self, name, rect, floor_color=None):
        self.name = name
        self.rect = rect
        self.floor_color = floor_color or COLORS['floor']

class Door:
    def __init__(self, x, y, width, height, orientation='horizontal'):
        self.rect = pygame.Rect(x, y, width, height)
        self.orientation = orientation

class Furniture:
    def __init__(self, name, rect, color, room):
        self.name = name
        self.rect = rect
        self.color = color
        self.room = room

class GameObject:
    def __init__(self, name, color, room):
        self.name = name
        self.color = color
        self.room = room
        self.pos = self.random_pos_in_room(room)

    def random_pos_in_room(self, room):
        # Keep objects away from walls and doors
        margin = 30
        x = random.randint(room.rect.x + margin, room.rect.x + room.rect.w - margin)
        y = random.randint(room.rect.y + margin, room.rect.y + room.rect.h - margin)
        return [x, y]

class Robot:
    def __init__(self, start_room, furniture_list=None):
        self.room = start_room
        self.pos = self.find_safe_spawn_position(start_room, furniture_list or [])
        self.has_object = None
    
    def find_safe_spawn_position(self, room, furniture_list):
        """Find a position in the room that doesn't overlap with furniture"""
        robot_radius = 15
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts:
            # Try random positions in the room
            margin = robot_radius + 20
            x = random.randint(room.rect.x + margin, room.rect.x + room.rect.width - margin)
            y = random.randint(room.rect.y + margin, room.rect.y + room.rect.height - margin)
            
            # Check if this position overlaps with any furniture
            robot_rect = pygame.Rect(x - robot_radius, y - robot_radius, robot_radius * 2, robot_radius * 2)
            
            position_is_safe = True
            for furniture in furniture_list:
                if furniture.room == room and robot_rect.colliderect(furniture.rect):
                    position_is_safe = False
                    break
            
            if position_is_safe:
                return [x, y]
            
            attempts += 1
        
        # Fallback: use room center if no safe position found
        return [room.rect.centerx, room.rect.centery]

# --- HELPER FUNCTIONS ---
def generate_furniture(rooms):
    """Generate appropriate furniture for each room type"""
    furniture = []
    
    # Furniture definitions by room type
    furniture_types = {
        'Living Room': [
            ('sofa', (120, 60), (139, 69, 19)),
            ('coffee_table', (60, 40), (160, 82, 45)),
            ('tv_stand', (80, 30), (70, 70, 70)),
            ('bookshelf', (25, 80), (101, 67, 33))
        ],
        'Kitchen': [
            ('counter', (100, 25), (200, 200, 200)),
            ('refrigerator', (30, 25), (240, 240, 240)),
            ('stove', (30, 25), (100, 100, 100)),
            ('sink', (40, 25), (192, 192, 192)),
            ('table', (60, 60), (160, 82, 45))
        ],
        'Bedroom': [
            ('bed', (80, 120), (255, 182, 193)),
            ('dresser', (60, 25), (139, 69, 19)),
            ('nightstand', (25, 25), (160, 82, 45)),
            ('wardrobe', (40, 25), (101, 67, 33))
        ],
        'Bathroom': [
            ('toilet', (25, 30), (255, 255, 255)),
            ('sink', (30, 20), (192, 192, 192)),
            ('bathtub', (60, 30), (255, 255, 255)),
            ('cabinet', (25, 20), (139, 69, 19))
        ]
    }
    
    for room in rooms:
        room_type = room.name
        if room_type in furniture_types:
            available_furniture = furniture_types[room_type]
            
            # Randomly select 2-4 pieces of furniture per room
            num_furniture = random.randint(2, min(4, len(available_furniture)))
            selected_furniture = random.sample(available_furniture, num_furniture)
            
            for furn_name, (width, height), color in selected_furniture:
                # Try to place furniture in the room (avoid doors and edges)
                attempts = 0
                placed = False
                
                while attempts < 20 and not placed:
                    margin = 35
                    x = random.randint(room.rect.x + margin, 
                                     max(room.rect.x + margin + 20, 
                                         room.rect.x + room.rect.width - width - margin))
                    y = random.randint(room.rect.y + margin,
                                     max(room.rect.y + margin + 20,
                                         room.rect.y + room.rect.height - height - margin))
                    
                    furn_rect = pygame.Rect(x, y, width, height)
                    
                    # Check if furniture overlaps with existing furniture
                    overlaps = False
                    for existing_furn in furniture:
                        if existing_furn.room == room and furn_rect.colliderect(existing_furn.rect):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        furniture.append(Furniture(furn_name, furn_rect, color, room))
                        placed = True
                    
                    attempts += 1
    
    return furniture

def draw_furniture(furniture_list):
    """Draw all furniture items"""
    for furniture in furniture_list:
        # Draw furniture base
        pygame.draw.rect(screen, furniture.color, furniture.rect)
        pygame.draw.rect(screen, (0, 0, 0), furniture.rect, 2)
        
        # Add furniture-specific details
        if furniture.name == 'sofa':
            # Add cushions
            cushion_width = furniture.rect.width // 3
            for i in range(3):
                cushion_x = furniture.rect.x + i * cushion_width
                pygame.draw.rect(screen, (120, 60, 15), 
                               (cushion_x + 2, furniture.rect.y + 2, cushion_width - 4, 20))
        
        elif furniture.name == 'bed':
            # Add pillow
            pygame.draw.rect(screen, (255, 255, 255), 
                           (furniture.rect.x + 5, furniture.rect.y + 5, 25, 15))
            # Add blanket pattern
            pygame.draw.rect(screen, (200, 150, 180), 
                           (furniture.rect.x + 5, furniture.rect.y + 25, 
                            furniture.rect.width - 10, furniture.rect.height - 30))
        
        elif furniture.name == 'tv_stand':
            # Add TV screen
            pygame.draw.rect(screen, (20, 20, 20), 
                           (furniture.rect.x + 10, furniture.rect.y - 20, 
                            furniture.rect.width - 20, 15))
        
        elif furniture.name == 'refrigerator':
            # Add door handle and freezer line
            pygame.draw.rect(screen, (200, 200, 200), 
                           (furniture.rect.x + furniture.rect.width - 5, 
                            furniture.rect.y + 10, 3, 8))
            pygame.draw.line(screen, (180, 180, 180), 
                           (furniture.rect.x, furniture.rect.y + furniture.rect.height // 3),
                           (furniture.rect.x + furniture.rect.width, furniture.rect.y + furniture.rect.height // 3), 2)
        
        elif furniture.name == 'stove':
            # Add burners
            for i in range(2):
                for j in range(2):
                    burner_x = furniture.rect.x + 5 + i * 10
                    burner_y = furniture.rect.y + 5 + j * 8
                    pygame.draw.circle(screen, (50, 50, 50), (burner_x, burner_y), 3)
        
        elif furniture.name == 'sink':
            # Add faucet
            pygame.draw.rect(screen, (169, 169, 169), 
                           (furniture.rect.centerx - 2, furniture.rect.y - 8, 4, 8))
        
        elif furniture.name == 'toilet':
            # Add toilet seat
            pygame.draw.ellipse(screen, (240, 240, 240), 
                              (furniture.rect.x + 2, furniture.rect.y + 2, 
                               furniture.rect.width - 4, furniture.rect.height - 10))
        
        elif furniture.name == 'bathtub':
            # Add bathtub interior
            pygame.draw.rect(screen, (240, 248, 255), 
                           (furniture.rect.x + 3, furniture.rect.y + 3, 
                            furniture.rect.width - 6, furniture.rect.height - 6))
        
        elif furniture.name == 'bookshelf':
            # Add shelf lines
            for i in range(1, 4):
                shelf_y = furniture.rect.y + (furniture.rect.height * i // 4)
                pygame.draw.line(screen, (80, 50, 20), 
                               (furniture.rect.x, shelf_y), 
                               (furniture.rect.x + furniture.rect.width, shelf_y), 2)
        
        elif furniture.name in ['dresser', 'nightstand', 'wardrobe', 'cabinet']:
            # Add drawer handles
            handle_y = furniture.rect.centery
            pygame.draw.circle(screen, (50, 50, 50), 
                             (furniture.rect.x + furniture.rect.width - 8, handle_y), 2)
        
        # Add furniture label (small text)
        if furniture.rect.width > 40:  # Only for larger furniture
            label_font = pygame.font.SysFont(None, 14)
            label = label_font.render(furniture.name.replace('_', ' ').title(), True, (50, 50, 50))
            label_x = furniture.rect.centerx - label.get_width() // 2
            label_y = furniture.rect.centery - label.get_height() // 2
            
            # Only draw label if it fits
            if (label_x > furniture.rect.x and 
                label_x + label.get_width() < furniture.rect.x + furniture.rect.width):
                screen.blit(label, (label_x, label_y))

def draw_house_structure(rooms, doors, furniture_list):
    # Fill background
    screen.fill((240, 240, 240))
    
    # Draw room floors
    for room in rooms:
        pygame.draw.rect(screen, room.floor_color, room.rect)
    
    # Draw furniture first (so it appears under other elements)
    draw_furniture(furniture_list)
    
    # Draw dynamic walls based on room layout
    wall_thickness = 12
    wall_color = (60, 60, 60)
    
    # Draw outer boundary
    boundary_margin = 50
    pygame.draw.rect(screen, wall_color, (boundary_margin, 80, SCREEN_W - 2*boundary_margin, wall_thickness))  # Top
    pygame.draw.rect(screen, wall_color, (boundary_margin, 80, wall_thickness, SCREEN_H - 140))  # Left
    pygame.draw.rect(screen, wall_color, (SCREEN_W - boundary_margin - wall_thickness, 80, wall_thickness, SCREEN_H - 140))  # Right
    pygame.draw.rect(screen, wall_color, (boundary_margin, SCREEN_H - 60, SCREEN_W - 2*boundary_margin, wall_thickness))  # Bottom
    
    # Draw walls between rooms (simplified - just room borders)
    for room in rooms:
        # Draw room borders but leave gaps for doors
        room_walls = []
        
        # Top wall
        room_walls.append(pygame.Rect(room.rect.x - 6, room.rect.y - 6, room.rect.width + 12, 6))
        # Left wall  
        room_walls.append(pygame.Rect(room.rect.x - 6, room.rect.y, 6, room.rect.height))
        # Right wall
        room_walls.append(pygame.Rect(room.rect.x + room.rect.width, room.rect.y, 6, room.rect.height))
        # Bottom wall
        room_walls.append(pygame.Rect(room.rect.x - 6, room.rect.y + room.rect.height, room.rect.width + 12, 6))
        
        for wall in room_walls:
            # Check if any door intersects this wall, if so, split the wall
            wall_segments = [wall]
            for door in doors:
                new_segments = []
                for segment in wall_segments:
                    if segment.colliderect(door.rect):
                        # Split wall around door
                        if wall.width > wall.height:  # Horizontal wall
                            if segment.x < door.rect.x:
                                new_segments.append(pygame.Rect(segment.x, segment.y, door.rect.x - segment.x, segment.height))
                            if door.rect.x + door.rect.width < segment.x + segment.width:
                                new_segments.append(pygame.Rect(door.rect.x + door.rect.width, segment.y, 
                                                              segment.x + segment.width - (door.rect.x + door.rect.width), segment.height))
                        else:  # Vertical wall
                            if segment.y < door.rect.y:
                                new_segments.append(pygame.Rect(segment.x, segment.y, segment.width, door.rect.y - segment.y))
                            if door.rect.y + door.rect.height < segment.y + segment.height:
                                new_segments.append(pygame.Rect(segment.x, door.rect.y + door.rect.height, segment.width,
                                                              segment.y + segment.height - (door.rect.y + door.rect.height)))
                    else:
                        new_segments.append(segment)
                wall_segments = new_segments
            
            # Draw the wall segments
            for segment in wall_segments:
                pygame.draw.rect(screen, wall_color, segment)
    
    # Draw doors with clear visual indicators
    for door in doors:
        # Draw door opening (bright color)
        pygame.draw.rect(screen, (255, 255, 255), door.rect)
        
        # Draw door frame (darker outline)
        pygame.draw.rect(screen, (100, 50, 0), door.rect, 3)
        
        # Add door swing indicator
        if door.orientation == 'horizontal':
            center_x = door.rect.centerx
            center_y = door.rect.y if door.rect.y > 200 else door.rect.y + door.rect.height
            pygame.draw.arc(screen, (150, 150, 150), 
                          (center_x - 20, center_y - 20, 40, 40), 0, 3.14, 2)
        else:
            center_x = door.rect.x if door.rect.x > 400 else door.rect.x + door.rect.width
            center_y = door.rect.centery
            pygame.draw.arc(screen, (150, 150, 150), 
                          (center_x - 20, center_y - 20, 40, 40), 0, 3.14, 2)
        
        # Add "DOOR" text label
        door_label = pygame.font.SysFont(None, 16).render("DOOR", True, (100, 50, 0))
        label_x = door.rect.centerx - door_label.get_width() // 2
        label_y = door.rect.centery - door_label.get_height() // 2
        
        label_bg = pygame.Rect(label_x - 2, label_y - 1, door_label.get_width() + 4, door_label.get_height() + 2)
        pygame.draw.rect(screen, (255, 255, 255), label_bg)
        screen.blit(door_label, (label_x, label_y))

def draw_room(room):
    # Draw room border (thicker)
    pygame.draw.rect(screen, (100, 100, 100), room.rect, 3)
    
    # Draw room label with bigger background
    label = font.render(room.name, True, (50, 50, 50))
    label_rect = label.get_rect()
    label_bg = pygame.Rect(room.rect.x + 15, room.rect.y + 15, label_rect.width + 16, label_rect.height + 10)
    pygame.draw.rect(screen, (255, 255, 255), label_bg)
    pygame.draw.rect(screen, (100, 100, 100), label_bg, 2)
    screen.blit(label, (room.rect.x + 23, room.rect.y + 20))

def draw_object(obj):
    # Draw object with shadow effect
    shadow_pos = (obj.pos[0] + 2, obj.pos[1] + 2)
    pygame.draw.circle(screen, (0, 0, 0, 50), shadow_pos, 12)
    pygame.draw.circle(screen, COLORS[obj.color], obj.pos, 10)
    pygame.draw.circle(screen, (255, 255, 255), (obj.pos[0] - 3, obj.pos[1] - 3), 3)
    
    # Object label
    label = font.render(obj.name, True, COLORS[obj.color])
    screen.blit(label, (obj.pos[0] + 15, obj.pos[1] - 10))

def draw_robot(robot):
    # Draw robot with shadow
    shadow_pos = (robot.pos[0] + 3, robot.pos[1] + 3)
    pygame.draw.circle(screen, (0, 0, 0, 80), shadow_pos, 18)
    pygame.draw.circle(screen, COLORS['green'], robot.pos, 15)
    pygame.draw.circle(screen, (150, 255, 150), robot.pos, 15, 3)
    
    # Robot "eyes"
    pygame.draw.circle(screen, (255, 255, 255), (robot.pos[0] - 5, robot.pos[1] - 5), 3)
    pygame.draw.circle(screen, (255, 255, 255), (robot.pos[0] + 5, robot.pos[1] - 5), 3)
    pygame.draw.circle(screen, (0, 0, 0), (robot.pos[0] - 5, robot.pos[1] - 5), 1)
    pygame.draw.circle(screen, (0, 0, 0), (robot.pos[0] + 5, robot.pos[1] - 5), 1)
    
    if robot.has_object:
        pygame.draw.circle(screen, COLORS[robot.has_object.color], (robot.pos[0]+20, robot.pos[1]), 8)

def move_robot(robot, dx, dy, rooms, doors, furniture_list):
    new_x = robot.pos[0] + dx
    new_y = robot.pos[1] + dy
    
    robot_radius = 15
    
    # Screen boundaries (generous to allow movement near walls)
    new_x = max(robot_radius + 55, min(SCREEN_W - robot_radius - 55, new_x))
    new_y = max(robot_radius + 85, min(SCREEN_H - robot_radius - 55, new_y))
    
    robot_rect = pygame.Rect(new_x - robot_radius, new_y - robot_radius, robot_radius * 2, robot_radius * 2)
    
    can_move = True
    
    # Check furniture collision (light collision - can squeeze past)
    for furniture in furniture_list:
        # Make furniture collision more forgiving
        furniture_collision_rect = pygame.Rect(furniture.rect.x + 5, furniture.rect.y + 5,
                                             furniture.rect.width - 10, furniture.rect.height - 10)
        if robot_rect.colliderect(furniture_collision_rect):
            can_move = False
            break
    
    if can_move:
        # Check outer house boundaries
        boundary_margin = 50
        outer_walls = [
            pygame.Rect(boundary_margin, 80, SCREEN_W - 2*boundary_margin, 12),  # Top
            pygame.Rect(boundary_margin, 80, 12, SCREEN_H - 140),  # Left
            pygame.Rect(SCREEN_W - boundary_margin - 12, 80, 12, SCREEN_H - 140),  # Right
            pygame.Rect(boundary_margin, SCREEN_H - 60, SCREEN_W - 2*boundary_margin, 12),  # Bottom
        ]
        
        for wall in outer_walls:
            if robot_rect.colliderect(wall):
                can_move = False
                break
        
        if can_move:
            # Check interior walls, but only the segments that don't have doors
            for room in rooms:
                # Get all potential wall segments for this room
                potential_walls = [
                    pygame.Rect(room.rect.x - 6, room.rect.y - 6, room.rect.width + 12, 6),  # Top
                    pygame.Rect(room.rect.x - 6, room.rect.y, 6, room.rect.height),  # Left
                    pygame.Rect(room.rect.x + room.rect.width, room.rect.y, 6, room.rect.height),  # Right
                    pygame.Rect(room.rect.x - 6, room.rect.y + room.rect.height, room.rect.width + 12, 6),  # Bottom
                ]
                
                for wall in potential_walls:
                    # Check if this wall has any doors cutting through it
                    wall_segments = [wall]
                    
                    # Split wall around any intersecting doors
                    for door in doors:
                        if wall.colliderect(door.rect):
                            new_segments = []
                            for segment in wall_segments:
                                if segment.colliderect(door.rect):
                                    # Split this segment around the door
                                    if segment.width > segment.height:  # Horizontal wall
                                        # Left part (before door)
                                        if segment.x < door.rect.x:
                                            left_part = pygame.Rect(segment.x, segment.y, 
                                                                  door.rect.x - segment.x, segment.height)
                                            if left_part.width > 5:  # Only add if significant size
                                                new_segments.append(left_part)
                                        
                                        # Right part (after door)
                                        door_end = door.rect.x + door.rect.width
                                        segment_end = segment.x + segment.width
                                        if door_end < segment_end:
                                            right_part = pygame.Rect(door_end, segment.y, 
                                                                    segment_end - door_end, segment.height)
                                            if right_part.width > 5:  # Only add if significant size
                                                new_segments.append(right_part)
                                    
                                    else:  # Vertical wall
                                        # Top part (before door)
                                        if segment.y < door.rect.y:
                                            top_part = pygame.Rect(segment.x, segment.y, 
                                                                 segment.width, door.rect.y - segment.y)
                                            if top_part.height > 5:  # Only add if significant size
                                                new_segments.append(top_part)
                                        
                                        # Bottom part (after door)
                                        door_end = door.rect.y + door.rect.height
                                        segment_end = segment.y + segment.height
                                        if door_end < segment_end:
                                            bottom_part = pygame.Rect(segment.x, door_end, 
                                                                     segment.width, segment_end - door_end)
                                            if bottom_part.height > 5:  # Only add if significant size
                                                new_segments.append(bottom_part)
                                else:
                                    # This segment doesn't intersect with door, keep it
                                    new_segments.append(segment)
                            wall_segments = new_segments
                    
                    # Check collision with the remaining wall segments (after door gaps)
                    for segment in wall_segments:
                        # Only check walls that separate different rooms or are exterior walls
                        is_separating_wall = False
                        
                        # Check if this wall segment is on the border between different rooms
                        for other_room in rooms:
                            if other_room != room:
                                # Expand other room slightly to detect adjacency
                                expanded = pygame.Rect(other_room.rect.x - 8, other_room.rect.y - 8, 
                                                     other_room.rect.width + 16, other_room.rect.height + 16)
                                if segment.colliderect(expanded):
                                    is_separating_wall = True
                                    break
                        
                        # Also consider walls on the house perimeter as separating walls
                        if (segment.x <= boundary_margin + 15 or 
                            segment.x + segment.width >= SCREEN_W - boundary_margin - 15 or
                            segment.y <= 90 or 
                            segment.y + segment.height >= SCREEN_H - 70):
                            is_separating_wall = True
                        
                        # Only block movement on separating walls
                        if is_separating_wall and robot_rect.colliderect(segment):
                            can_move = False
                            break
                    
                    if not can_move:
                        break
                
                if not can_move:
                    break
    
    if can_move:
        robot.pos[0] = new_x
        robot.pos[1] = new_y

def point_in_rect(point, rect):
    return rect.x < point[0] < rect.x+rect.w and rect.y < point[1] < rect.y+rect.h

def get_room_for_pos(pos, rooms):
    for room in rooms:
        if point_in_rect(pos, room.rect):
            return room
    return None

def draw_hud(score, elapsed_time, level):
    # Draw HUD background
    hud_rect = pygame.Rect(0, 0, SCREEN_W, 50)
    pygame.draw.rect(screen, (40, 40, 40), hud_rect)
    pygame.draw.rect(screen, (100, 100, 100), hud_rect, 2)
    
    # Draw score, time, and level
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    time_text = font.render(f"Time: {elapsed_time:.1f}s", True, (255, 255, 255))
    level_text = font.render(f"Level: {level}", True, (255, 255, 255))
    
    screen.blit(score_text, (15, 15))
    screen.blit(time_text, (150, 15))
    screen.blit(level_text, (300, 15))

def draw_completion_message(score_gained, elapsed_time, level):
    """Draw a celebratory completion message"""
    # Create overlay
    overlay = pygame.Surface((SCREEN_W, SCREEN_H))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Main completion message
    big_font = pygame.font.SysFont(None, 48)
    success_text = big_font.render("🎉 TASK COMPLETED! 🎉", True, (0, 255, 0))
    success_rect = success_text.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 60))
    screen.blit(success_text, success_rect)
    
    # Score details
    medium_font = pygame.font.SysFont(None, 32)
    score_text = medium_font.render(f"Points Earned: +{score_gained}", True, (255, 255, 255))
    score_rect = score_text.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 10))
    screen.blit(score_text, score_rect)
    
    # Time details
    time_text = medium_font.render(f"Completion Time: {elapsed_time:.1f}s", True, (255, 255, 255))
    time_rect = time_text.get_rect(center=(SCREEN_W//2, SCREEN_H//2 + 20))
    screen.blit(time_text, time_rect)
    
    # Level info
    level_text = medium_font.render(f"Moving to Level {level + 1}", True, (255, 255, 0))
    level_rect = level_text.get_rect(center=(SCREEN_W//2, SCREEN_H//2 + 50))
    screen.blit(level_text, level_rect)
    
    # Continue instruction
    continue_text = font.render("Press any key to continue...", True, (200, 200, 200))
    continue_rect = continue_text.get_rect(center=(SCREEN_W//2, SCREEN_H//2 + 90))
    screen.blit(continue_text, continue_rect)
    
    # Add some decorative elements
    # Draw stars around the success message
    star_positions = [
        (SCREEN_W//2 - 150, SCREEN_H//2 - 80),
        (SCREEN_W//2 + 150, SCREEN_H//2 - 80),
        (SCREEN_W//2 - 200, SCREEN_H//2 - 40),
        (SCREEN_W//2 + 200, SCREEN_H//2 - 40),
    ]
    
    for star_x, star_y in star_positions:
        # Draw simple star
        star_points = []
        for i in range(10):
            angle = i * 3.14159 / 5
            radius = 15 if i % 2 == 0 else 8
            x = star_x + radius * 0.8 * (1 if i % 4 < 2 else -1)
            y = star_y + radius * 0.8 * (1 if i % 4 == 0 or i % 4 == 3 else -1)
            star_points.append((x, y))
        
        if len(star_points) >= 3:
            pygame.draw.polygon(screen, (255, 255, 0), star_points[:8])

# --- GAME LOGIC ---
def generate_floor_plan():
    # Create different random house layouts
    rooms = []
    doors = []
    
    # Choose a random layout type
    layout_types = ['traditional', 'modern', 'compact', 'open_plan']
    layout = random.choice(layout_types)
    
    if layout == 'traditional':
        # Traditional 2x2 grid layout
        room_names = random.sample(ROOMS, 4)  # Random room assignment
        w, h = (SCREEN_W - 120) // 2, (SCREEN_H - 140) // 2
        
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        random.shuffle(positions)  # Random room positions
        
        colors = [(220, 200, 160), (200, 220, 180), (180, 200, 220), (200, 200, 200)]
        random.shuffle(colors)
        
        for i, (col, row) in enumerate(positions):
            x = 60 + col * w
            y = 90 + row * h
            room = Room(room_names[i], pygame.Rect(x + 6, y + 6, w - 12, h - 12), colors[i])
            rooms.append(room)
        
        # Ensure connectivity by adding doors systematically
        wall_thickness = 12
        
        # Always add at least one horizontal door to connect top/bottom rows
        door_x = random.randint(100, 140)
        doors.append(Door(door_x, 90 + h - 6, 40, wall_thickness, 'horizontal'))
        
        # Always add at least one vertical door to connect left/right columns
        door_y = random.randint(120, 160)
        doors.append(Door(60 + w - 6, door_y, wall_thickness, 40, 'vertical'))
        
        # Optionally add more doors for variety (but ensure minimum connectivity)
        if random.choice([True, False]):
            door_y2 = random.randint(90 + h + 20, 90 + h + 60)
            doors.append(Door(60 + w - 6, door_y2, wall_thickness, 40, 'vertical'))
        
        if random.choice([True, False]):
            door_x2 = random.randint(60 + w + 20, 60 + w + 60)
            doors.append(Door(door_x2, 90 + h - 6, 40, wall_thickness, 'horizontal'))
            
    elif layout == 'modern':
        # Modern L-shaped layout
        room_names = random.sample(ROOMS, 4)
        colors = [(220, 200, 160), (200, 220, 180), (180, 200, 220), (200, 200, 200)]
        random.shuffle(colors)
        
        # Large living area
        rooms.append(Room(room_names[0], pygame.Rect(66, 96, 300, 200), colors[0]))
        # Kitchen
        rooms.append(Room(room_names[1], pygame.Rect(376, 96, 200, 120), colors[1]))
        # Bedroom
        rooms.append(Room(room_names[2], pygame.Rect(376, 226, 200, 150), colors[2]))
        # Bathroom
        rooms.append(Room(room_names[3], pygame.Rect(586, 226, 150, 150), colors[3]))
        
        # Ensure all rooms are connected
        doors.append(Door(366, random.randint(120, 160), 12, 40, 'vertical'))    # Living room to kitchen - REQUIRED
        doors.append(Door(366, random.randint(240, 280), 12, 40, 'vertical'))    # Living room to bedroom - REQUIRED
        doors.append(Door(random.randint(420, 460), 216, 40, 12, 'horizontal'))  # Kitchen to bedroom - REQUIRED
        doors.append(Door(576, random.randint(260, 300), 12, 40, 'vertical'))    # Bedroom to bathroom - REQUIRED
        
    elif layout == 'compact':
        # Compact apartment style
        room_names = random.sample(ROOMS, 4)
        colors = [(220, 200, 160), (200, 220, 180), (180, 200, 220), (200, 200, 200)]
        random.shuffle(colors)
        
        # Vertical arrangement
        room_height = (SCREEN_H - 140) // 4
        for i in range(4):
            y = 90 + i * room_height
            width = random.randint(250, 350)
            x = random.randint(60, SCREEN_W - width - 60)
            rooms.append(Room(room_names[i], pygame.Rect(x + 6, y + 6, width - 12, room_height - 12), colors[i]))
        
        # Connect ALL adjacent rooms with doors (ensure full connectivity)
        for i in range(3):
            room1, room2 = rooms[i], rooms[i + 1]
            # Find overlapping x area
            overlap_start = max(room1.rect.x, room2.rect.x)
            overlap_end = min(room1.rect.x + room1.rect.width, room2.rect.x + room2.rect.width)
            
            # Check if we have enough overlap for a door with margins
            required_width = 80  # 20 margin + 40 door + 20 margin
            if overlap_end > overlap_start + required_width:
                # Sufficient overlap - place door in overlapping area
                door_x = random.randint(overlap_start + 20, overlap_end - 60)
                door_y = room1.rect.y + room1.rect.height - 6
                doors.append(Door(door_x, door_y, 40, 12, 'horizontal'))
            else:
                # Insufficient overlap - use alternative door placement
                # Place door at the larger room's boundary, ensuring it's within both rooms' bounds
                room1_center_x = room1.rect.x + room1.rect.width // 2
                room2_center_x = room2.rect.x + room2.rect.width // 2
                
                # Find a safe x position that works for both rooms
                safe_start = max(room1.rect.x + 20, room2.rect.x + 20)
                safe_end = min(room1.rect.x + room1.rect.width - 60, room2.rect.x + room2.rect.width - 60)
                
                if safe_end > safe_start:
                    door_x = random.randint(safe_start, safe_end)
                else:
                    # Last resort - use the overlap center, clamped to valid range
                    door_x = max(safe_start, min(safe_end, (overlap_start + overlap_end) // 2 - 20))
                    door_x = max(room1.rect.x + 10, min(room1.rect.x + room1.rect.width - 50, door_x))
                    door_x = max(room2.rect.x + 10, min(room2.rect.x + room2.rect.width - 50, door_x))
                
                door_y = room1.rect.y + room1.rect.height - 6
                doors.append(Door(door_x, door_y, 40, 12, 'horizontal'))
    
    else:  # open_plan
        # Open plan with fewer walls but guaranteed connectivity
        room_names = random.sample(ROOMS, 3)  # Only 3 rooms for open plan
        colors = [(220, 200, 160), (200, 220, 180), (180, 200, 220)]
        random.shuffle(colors)
        
        # Large open area
        rooms.append(Room(room_names[0], pygame.Rect(66, 96, 400, 300), colors[0]))
        # Side room 1
        rooms.append(Room(room_names[1], pygame.Rect(476, 96, 200, 150), colors[1]))
        # Side room 2  
        rooms.append(Room(room_names[2], pygame.Rect(476, 256, 200, 140), colors[2]))
        
        # Ensure both side rooms connect to main area
        doors.append(Door(466, random.randint(130, 170), 12, 40, 'vertical'))    # Main to room 1 - REQUIRED
        doors.append(Door(466, random.randint(280, 320), 12, 40, 'vertical'))    # Main to room 2 - REQUIRED
    
    # Validation: Ensure all rooms are reachable
    doors = validate_connectivity(rooms, doors)
    
    return rooms, doors

def validate_connectivity(rooms, doors):
    """Ensure all rooms are connected and reachable"""
    if len(rooms) <= 1:
        return doors
    
    # Build adjacency list from existing doors
    adjacency = {i: set() for i in range(len(rooms))}
    
    for door in doors:
        # Find which rooms this door connects
        connected_rooms = []
        for i, room in enumerate(rooms):
            # Check if door is on the border of this room
            room_border = pygame.Rect(room.rect.x - 10, room.rect.y - 10, 
                                    room.rect.width + 20, room.rect.height + 20)
            if door.rect.colliderect(room_border):
                connected_rooms.append(i)
        
        # Connect the rooms that share this door
        if len(connected_rooms) >= 2:
            for i in range(len(connected_rooms)):
                for j in range(i + 1, len(connected_rooms)):
                    adjacency[connected_rooms[i]].add(connected_rooms[j])
                    adjacency[connected_rooms[j]].add(connected_rooms[i])
    
    # Find connected components using DFS
    visited = [False] * len(rooms)
    components = []
    
    def dfs(node, component):
        visited[node] = True
        component.append(node)
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                dfs(neighbor, component)
    
    for i in range(len(rooms)):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(component)
    
    # If we have multiple components, connect them
    while len(components) > 1:
        # Connect the two largest components
        comp1 = max(components, key=len)
        components.remove(comp1)
        comp2 = max(components, key=len)
        components.remove(comp2)
        
        # Find the best rooms to connect
        best_distance = float('inf')
        best_room1, best_room2 = None, None
        
        for r1 in comp1:
            for r2 in comp2:
                # Calculate distance between room centers
                room1, room2 = rooms[r1], rooms[r2]
                dist = ((room1.rect.centerx - room2.rect.centerx) ** 2 + 
                       (room1.rect.centery - room2.rect.centery) ** 2) ** 0.5
                if dist < best_distance:
                    best_distance = dist
                    best_room1, best_room2 = r1, r2
        
        # Create a door connecting these rooms
        if best_room1 is not None and best_room2 is not None:
            room1, room2 = rooms[best_room1], rooms[best_room2]
            
            # Determine door position and orientation
            if abs(room1.rect.centerx - room2.rect.centerx) > abs(room1.rect.centery - room2.rect.centery):
                # Horizontal separation - create vertical door
                if room1.rect.centerx < room2.rect.centerx:
                    # Room1 is left of room2
                    door_x = room1.rect.x + room1.rect.width
                    door_y = max(room1.rect.y + 20, room2.rect.y + 20)
                    door_y = min(door_y, room1.rect.y + room1.rect.height - 60, room2.rect.y + room2.rect.height - 60)
                else:
                    # Room2 is left of room1
                    door_x = room2.rect.x + room2.rect.width
                    door_y = max(room1.rect.y + 20, room2.rect.y + 20)
                    door_y = min(door_y, room1.rect.y + room1.rect.height - 60, room2.rect.y + room2.rect.height - 60)
                
                doors.append(Door(door_x - 6, door_y, 12, 40, 'vertical'))
            else:
                # Vertical separation - create horizontal door
                if room1.rect.centery < room2.rect.centery:
                    # Room1 is above room2
                    door_x = max(room1.rect.x + 20, room2.rect.x + 20)
                    door_x = min(door_x, room1.rect.x + room1.rect.width - 60, room2.rect.x + room2.rect.width - 60)
                    door_y = room1.rect.y + room1.rect.height
                else:
                    # Room2 is above room1
                    door_x = max(room1.rect.x + 20, room2.rect.x + 20)
                    door_x = min(door_x, room1.rect.x + room1.rect.width - 60, room2.rect.x + room2.rect.width - 60)
                    door_y = room2.rect.y + room2.rect.height
                
                doors.append(Door(door_x, door_y - 6, 40, 12, 'horizontal'))
        
        # Merge components
        merged_component = comp1 + comp2
        components.append(merged_component)
        
        # Update adjacency
        if best_room1 is not None and best_room2 is not None:
            adjacency[best_room1].add(best_room2)
            adjacency[best_room2].add(best_room1)
    
    return doors

def generate_objects(rooms, level):
    """Generate more objects with increasing difficulty"""
    objects = []
    
    # Base number of objects increases with level
    base_objects = 2 + level // 2  # Start with 2, increase every 2 levels
    distractor_objects = level // 3  # Add distractors as level increases
    total_objects = min(base_objects + distractor_objects, 12)  # Cap at 12 objects
    
    for i in range(total_objects):
        # Random object type and color
        obj_type = random.choice(OBJECT_TYPES)
        obj_color = random.choice(OBJECT_COLORS)
        obj_name = f"{obj_color} {obj_type}"
        
        # Place in random room
        room = random.choice(rooms)
        obj = GameObject(obj_name, obj_color, room)
        objects.append(obj)
    
    return objects

def generate_task(objects, rooms, furniture, level):
    """Generate diverse and challenging tasks"""
    task_types = [
        'room_delivery',      # Classic: pick up X, drop in Y room
        'furniture_proximity', # New: drop X near Y furniture  
        'furniture_specific',  # New: drop X on/near specific furniture type
        'room_collection',    # New: collect X from Y room, drop in Z room
        'color_matching',     # New: collect all objects of X color
        'multi_step',         # New: pick up X, then Y, drop both in Z
    ]
    
    # Weight task types based on level (harder tasks at higher levels)
    if level <= 2:
        task_type = random.choice(['room_delivery', 'furniture_proximity'])
    elif level <= 5:
        task_type = random.choice(['room_delivery', 'furniture_proximity', 'furniture_specific', 'room_collection'])
    else:
        task_type = random.choice(task_types)
    
    if task_type == 'room_delivery':
        # Classic room delivery
        obj = random.choice(objects)
        target_room = random.choice([r for r in rooms if r != obj.room])
        return {
            'type': 'room_delivery',
            'object': obj,
            'target_room': target_room,
            'instruction': f'Pick up the {obj.name} and drop it in the {target_room.name}.'
        }
    
    elif task_type == 'furniture_proximity':
        # Drop near specific furniture
        obj = random.choice(objects)
        target_room = random.choice([r for r in rooms if r != obj.room])
        room_furniture = [f for f in furniture if f.room == target_room]
        
        if room_furniture:
            target_furniture = random.choice(room_furniture)
            return {
                'type': 'furniture_proximity',
                'object': obj,
                'target_room': target_room,
                'target_furniture': target_furniture,
                'instruction': f'Pick up the {obj.name} and drop it near the {target_furniture.name.replace("_", " ")} in the {target_room.name}.'
            }
    
    elif task_type == 'furniture_specific':
        # Drop on specific furniture type
        obj = random.choice(objects)
        furniture_types = ['table', 'counter', 'dresser', 'nightstand', 'bookshelf']
        target_furn_type = random.choice(furniture_types)
        
        # Find furniture of this type
        matching_furniture = [f for f in furniture if target_furn_type in f.name]
        if matching_furniture:
            target_furniture = random.choice(matching_furniture)
            return {
                'type': 'furniture_specific',
                'object': obj,
                'target_furniture': target_furniture,
                'target_room': target_furniture.room,
                'instruction': f'Pick up the {obj.name} and place it on the {target_furniture.name.replace("_", " ")}.'
            }
    
    elif task_type == 'room_collection':
        # Collect from specific room, deliver to another
        source_room = random.choice(rooms)
        target_room = random.choice([r for r in rooms if r != source_room])
        source_objects = [o for o in objects if o.room == source_room]
        
        if source_objects:
            obj = random.choice(source_objects)
            return {
                'type': 'room_collection',
                'object': obj,
                'source_room': source_room,
                'target_room': target_room,
                'instruction': f'Find the {obj.name} in the {source_room.name} and bring it to the {target_room.name}.'
            }
    
    elif task_type == 'color_matching':
        # Collect objects of specific color
        if len(objects) >= 3:
            available_colors = list(set(obj.color for obj in objects))
            if len(available_colors) >= 2:
                target_color = random.choice(available_colors)
                colored_objects = [o for o in objects if o.color == target_color]
                if len(colored_objects) >= 2:
                    target_room = random.choice(rooms)
                    return {
                        'type': 'color_matching',
                        'target_color': target_color,
                        'target_room': target_room,
                        'objects': colored_objects,
                        'instruction': f'Collect all {target_color} objects and put them in the {target_room.name}.'
                    }
    
    elif task_type == 'multi_step':
        # Multi-step task
        if len(objects) >= 2:
            obj1, obj2 = random.sample(objects, 2)
            target_room = random.choice(rooms)
            return {
                'type': 'multi_step',
                'objects': [obj1, obj2],
                'target_room': target_room,
                'collected': [],
                'instruction': f'Collect the {obj1.name} and {obj2.name}, then bring both to the {target_room.name}.'
            }
    
    # Fallback to simple room delivery
    obj = random.choice(objects)
    target_room = random.choice([r for r in rooms if r != obj.room])
    return {
        'type': 'room_delivery',
        'object': obj,
        'target_room': target_room,
        'instruction': f'Pick up the {obj.name} and drop it in the {target_room.name}.'
    }

def check_task_completion(robot, task, rooms):
    """Check completion for different task types"""
    if task['type'] == 'room_delivery':
        return (
            robot.has_object is None and
            get_room_for_pos(robot.pos, rooms) == task['target_room'] and
            task['object'].room == task['target_room']
        )
    
    elif task['type'] in ['furniture_proximity', 'furniture_specific']:
        if robot.has_object is not None:
            return False
        
        # Check if object is in target room
        if task['object'].room != task['target_room']:
            return False
        
        # Check if object is near target furniture
        obj_pos = task['object'].pos
        furniture_rect = task['target_furniture'].rect
        
        # Consider "near" as within 50 pixels
        distance = ((obj_pos[0] - furniture_rect.centerx) ** 2 + 
                   (obj_pos[1] - furniture_rect.centery) ** 2) ** 0.5
        return distance <= 70
    
    elif task['type'] == 'room_collection':
        return (
            robot.has_object is None and
            get_room_for_pos(robot.pos, rooms) == task['target_room'] and
            task['object'].room == task['target_room']
        )
    
    elif task['type'] == 'color_matching':
        if robot.has_object is not None:
            return False
        
        # Check if all objects of target color are in target room
        for obj in task['objects']:
            if obj.room != task['target_room']:
                return False
        return True
    
    elif task['type'] == 'multi_step':
        if robot.has_object is not None:
            return False
        
        # Check if both objects are in target room
        for obj in task['objects']:
            if obj.room != task['target_room']:
                return False
        return True
    
    return False

# --- MAIN LOOP ---
def main():
    level = 1
    score = 0
    
    while True:
        # Setup new game
        global rooms, doors, furniture
        rooms, doors = generate_floor_plan()
        furniture = generate_furniture(rooms)  # Generate furniture once per level
        objects = generate_objects(rooms, level)
        robot = Robot(random.choice(rooms), furniture)
        task = generate_task(objects, rooms, furniture, level)
        start_time = time.time()
        message = ""
        running = True

        while running:
            # Handle continuous movement
            keys = pygame.key.get_pressed()
            move_speed = 2
            if keys[pygame.K_LEFT]: move_robot(robot, -move_speed, 0, rooms, doors, furniture)
            if keys[pygame.K_RIGHT]: move_robot(robot, move_speed, 0, rooms, doors, furniture)
            if keys[pygame.K_UP]: move_robot(robot, 0, -move_speed, rooms, doors, furniture)
            if keys[pygame.K_DOWN]: move_robot(robot, 0, move_speed, rooms, doors, furniture)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Draw everything
            draw_house_structure(rooms, doors, furniture)
            
            # Draw HUD (score, time, level)
            draw_hud(score, elapsed_time, level)
            
            # Draw rooms, objects, robot
            for room in rooms:
                draw_room(room)
            for obj in objects:
                if not (robot.has_object == obj):
                    draw_object(obj)
            draw_robot(robot)

            # Instructions
            instr = font.render("Task: " + task['instruction'], True, (255,255,255))
            instr_bg = pygame.Rect(10, SCREEN_H - 80, instr.get_width() + 10, 25)
            pygame.draw.rect(screen, (40, 40, 40), instr_bg)
            screen.blit(instr, (15, SCREEN_H - 75))
            
            # Controls info
            controls = font.render("Controls: Arrow keys to move, P to pick up, D to drop", True, (200,200,200))
            controls_bg = pygame.Rect(10, SCREEN_H - 50, controls.get_width() + 10, 20)
            pygame.draw.rect(screen, (40, 40, 40), controls_bg)
            screen.blit(controls, (15, SCREEN_H - 47))
            
            if message:
                msg = font.render(message, True, (255,100,100))
                msg_bg = pygame.Rect(10, SCREEN_H - 25, msg.get_width() + 10, 20)
                pygame.draw.rect(screen, (40, 40, 40), msg_bg)
                screen.blit(msg, (15, SCREEN_H - 22))
            
            pygame.display.flip()
            clock.tick(60)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    # Pick up
                    if event.key == pygame.K_p and not robot.has_object:
                        for obj in objects:
                            if abs(robot.pos[0] - obj.pos[0]) < 25 and abs(robot.pos[1] - obj.pos[1]) < 25:
                                robot.has_object = obj
                                message = f"Picked up {obj.name}"
                                break
                    # Drop
                    if event.key == pygame.K_d and robot.has_object:
                        cur_room = get_room_for_pos(robot.pos, rooms)
                        if cur_room:
                            robot.has_object.pos = robot.pos.copy()
                            robot.has_object.room = cur_room
                            message = f"Dropped {robot.has_object.name} in {cur_room.name}"
                            robot.has_object = None
                            
            # Check win/fail
            if check_task_completion(robot, task, rooms):
                elapsed = time.time() - start_time
                time_bonus = max(0, 100 - int(elapsed))
                level_bonus = level * 50
                task_score = 100 + time_bonus + level_bonus
                score += task_score
                
                draw_completion_message(task_score, elapsed, level)
                pygame.display.flip()
                pygame.time.wait(1000)
                wait = True
                while wait:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            level += 1
                            wait = False
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                break
                
            if elapsed_time > 60:
                message = "Time's up! Press any key to restart level."
                pygame.display.flip()
                pygame.time.wait(1000)
                wait = True
                while wait:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            wait = False
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                break

if __name__ == "__main__":
    main()

