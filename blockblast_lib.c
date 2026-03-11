#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include "raylib.h"

#define BOARD_SIZE 8
#define MAX_SHAPE_SIZE 5

typedef struct {
    int width;
    int height;
    int blocks[MAX_SHAPE_SIZE][MAX_SHAPE_SIZE];
    Color color;
} Shape;

Shape shapes_pool[] = {
    {1, 1, {{1}}, RED},
    {2, 2, {{1,1}, {1,1}}, ORANGE},
    {3, 3, {{1,1,1}, {1,1,1}, {1,1,1}}, YELLOW},
    {3, 1, {{1,1,1}}, GREEN},
    {1, 3, {{1},{1},{1}}, BLUE},
    {4, 1, {{1,1,1,1}}, PURPLE},
    {1, 4, {{1},{1},{1},{1}}, VIOLET},
    {5, 1, {{1,1,1,1,1}}, DARKBLUE},
    {1, 5, {{1},{1},{1},{1},{1}}, DARKGREEN},
    {2, 2, {{1,0}, {1,1}}, LIME},
    {2, 2, {{0,1}, {1,1}}, GOLD},
    {2, 2, {{1,1}, {1,0}}, PINK},
    {2, 2, {{1,1}, {0,1}}, SKYBLUE},
    {3, 3, {{1,0,0}, {1,0,0}, {1,1,1}}, BEIGE},
    {3, 3, {{0,0,1}, {0,0,1}, {1,1,1}}, BROWN},
    {3, 2, {{1,1,1}, {0,1,0}}, DARKGRAY},
    {2, 3, {{1,0}, {1,1}, {1,0}}, MAROON}
};
int num_shapes_pool = 17;

typedef struct {
    int board[BOARD_SIZE][BOARD_SIZE];
    int board_colors[BOARD_SIZE][BOARD_SIZE]; // Store color index (0-16) or -1
    int current_shapes[3]; // Indices in shapes_pool
    bool shape_active[3];
    int score;
    bool game_over;
} GameState;

void generate_shapes(GameState* state) {
    for (int i = 0; i < 3; i++) {
        state->current_shapes[i] = rand() % num_shapes_pool;
        state->shape_active[i] = true;
    }
}

GameState* init_game() {
    GameState* state = (GameState*)malloc(sizeof(GameState));
    memset(state, 0, sizeof(GameState));
    for (int i=0; i<8; i++)
        for (int j=0; j<8; j++)
            state->board_colors[i][j] = -1;
    generate_shapes(state);
    return state;
}

void reset_game(GameState* state) {
    memset(state, 0, sizeof(GameState));
    for (int i=0; i<8; i++)
        for (int j=0; j<8; j++)
            state->board_colors[i][j] = -1;
    generate_shapes(state);
}

bool can_place(GameState* state, int shape_idx, int row, int col) {
    if (shape_idx < 0 || shape_idx > 2 || !state->shape_active[shape_idx]) return false;
    Shape s = shapes_pool[state->current_shapes[shape_idx]];
    if (row < 0 || col < 0 || row + s.height > BOARD_SIZE || col + s.width > BOARD_SIZE) return false;
    
    for (int r = 0; r < s.height; r++) {
        for (int c = 0; c < s.width; c++) {
            if (s.blocks[r][c] && state->board[row + r][col + c]) return false;
        }
    }
    return true;
}

int clear_lines(GameState* state) {
    bool rows_to_clear[BOARD_SIZE] = {false};
    bool cols_to_clear[BOARD_SIZE] = {false};
    int lines = 0;

    for (int i = 0; i < BOARD_SIZE; i++) {
        bool r_full = true, c_full = true;
        for (int j = 0; j < BOARD_SIZE; j++) {
            if (!state->board[i][j]) r_full = false;
            if (!state->board[j][i]) c_full = false;
        }
        rows_to_clear[i] = r_full;
        cols_to_clear[i] = c_full;
    }

    for (int i = 0; i < BOARD_SIZE; i++) {
        if (rows_to_clear[i]) {
            lines++;
            for (int j = 0; j < BOARD_SIZE; j++) {
                state->board[i][j] = 0;
                state->board_colors[i][j] = -1;
            }
        }
    }
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (cols_to_clear[i]) {
            lines++;
            for (int j = 0; j < BOARD_SIZE; j++) {
                state->board[j][i] = 0;
                state->board_colors[j][i] = -1;
            }
        }
    }
    return lines;
}

bool check_game_over(GameState* state) {
    for (int i = 0; i < 3; i++) {
        if (!state->shape_active[i]) continue;
        for (int r = 0; r < BOARD_SIZE; r++) {
            for (int c = 0; c < BOARD_SIZE; c++) {
                if (can_place(state, i, r, c)) return false;
            }
        }
    }
    return true;
}

// Fills obs array with state. Size should be 8*8 + 3*5*5 = 64 + 75 = 139.
// Board (64) + 3 Shapes (5x5 each = 75)
void get_observation(GameState* state, int* obs) {
    int idx = 0;
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            obs[idx++] = state->board[i][j];
        }
    }
    for (int i=0; i<3; i++) {
        Shape s = shapes_pool[state->current_shapes[i]];
        for (int r=0; r<5; r++) {
            for (int c=0; c<5; c++) {
                if (state->shape_active[i] && r < s.height && c < s.width) {
                    obs[idx++] = s.blocks[r][c];
                } else {
                    obs[idx++] = 0;
                }
            }
        }
    }
}

// Action mask array of size 192 (3 shapes * 8 rows * 8 cols)
void get_action_mask(GameState* state, int* mask) {
    for (int i = 0; i < 192; i++) {
        int shape_idx = i / 64;
        int row = (i / 8) % 8;
        int col = i % 8;
        mask[i] = can_place(state, shape_idx, row, col) ? 1 : 0;
    }
}

void step_game(GameState* state, int action, float* reward, bool* done) {
    if (state->game_over) {
        *reward = 0;
        *done = true;
        return;
    }

    int shape_idx = action / 64;
    int row = (action / 8) % 8;
    int col = action % 8;

    if (!can_place(state, shape_idx, row, col)) {
        *reward = -10.0f; // Invalid action penalty
        *done = true;
        state->game_over = true;
        return;
    }

    Shape s = shapes_pool[state->current_shapes[shape_idx]];
    int blocks_placed = 0;
    for (int r = 0; r < s.height; r++) {
        for (int c = 0; c < s.width; c++) {
            if (s.blocks[r][c]) {
                state->board[row + r][col + c] = 1;
                state->board_colors[row + r][col + c] = state->current_shapes[shape_idx];
                blocks_placed++;
            }
        }
    }
    state->shape_active[shape_idx] = false;
    
    int lines_cleared = clear_lines(state);
    
    float step_reward = blocks_placed + (lines_cleared * 10.0f) + 1.0f; // slight bonus for surviving
    state->score += step_reward;
    *reward = step_reward;

    bool all_used = true;
    for (int i=0; i<3; i++) if (state->shape_active[i]) all_used = false;
    if (all_used) generate_shapes(state);

    *done = check_game_over(state);
    state->game_over = *done;
}

void free_game(GameState* state) {
    free(state);
}

// Rendering
bool window_initialized = false;
void init_render() {
    if (!window_initialized) {
        InitWindow(500, 700, "Blockblast RL Training");
        SetTargetFPS(60);
        window_initialized = true;
    }
}

void render_game_state(GameState* state) {
    if (!window_initialized) init_render();
    
    BeginDrawing();
    ClearBackground(RAYWHITE);
    
    int cell_size = 50;
    int offset_x = 50, offset_y = 50;

    // Draw Board
    for (int i=0; i<8; i++) {
        for (int j=0; j<8; j++) {
            Rectangle r = { offset_x + j*cell_size, offset_y + i*cell_size, cell_size-2, cell_size-2 };
            Color c = LIGHTGRAY;
            if (state->board[i][j]) {
                int c_idx = state->board_colors[i][j];
                if (c_idx >= 0 && c_idx < num_shapes_pool) c = shapes_pool[c_idx].color;
                else c = DARKGRAY;
            }
            DrawRectangleRec(r, c);
        }
    }
    
    DrawText(TextFormat("Score: %d", state->score), 20, 10, 20, BLACK);
    if (state->game_over) DrawText("GAME OVER", 150, 300, 40, RED);

    // Draw Shapes
    for (int i=0; i<3; i++) {
        if (!state->shape_active[i]) continue;
        Shape s = shapes_pool[state->current_shapes[i]];
        for (int r=0; r<s.height; r++) {
            for (int c=0; c<s.width; c++) {
                if (s.blocks[r][c]) {
                    DrawRectangle(75 + i*150 + c*cell_size*0.6, 500 + r*cell_size*0.6, cell_size*0.6 - 1, cell_size*0.6 - 1, s.color);
                }
            }
        }
    }
    
    EndDrawing();
}

void close_render() {
    if (window_initialized) {
        CloseWindow();
        window_initialized = false;
    }
}
