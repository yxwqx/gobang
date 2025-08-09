import requests

def calculate_position(position):
    col_idx = ord(position[0]) - ord('a')
    row_idx = ord(position[1]) - ord('a')
    return row_idx * 15 + col_idx

def extract_directions(position, chess_state):
    directions = ['', '', '', '']
    center_col = ord(position[0]) - ord('a')
    center_row = ord(position[1]) - ord('a')
    for offset in range(-4, 5):
        row = center_row
        col = center_col + offset
        if 0 <= row < 15 and 0 <= col < 15:
            idx = row * 15 + col
            directions[0] += chess_state[idx]
        else:
            directions[0] += ' '
        row = center_row + offset
        col = center_col
        if 0 <= row < 15 and 0 <= col < 15:
            idx = row * 15 + col
            directions[2] += chess_state[idx]
        else:
            directions[2] += ' '
        row = center_row + offset
        col = center_col + offset
        if 0 <= row < 15 and 0 <= col < 15:
            idx = row * 15 + col
            directions[1] += chess_state[idx]
        else:
            directions[1] += ' '
        row = center_row - offset
        col = center_col + offset
        if 0 <= row < 15 and 0 <= col < 15:
            idx = row * 15 + col
            directions[3] += chess_state[idx]
        else:
            directions[3] += ' '
    return directions

response = requests.get("http://183.175.12.27:8004/step_07/")
data = response.json()
move_sequence = data['board']
target_positions = data['coord']

chess_board = '.' * (15 * 15)

current_turn = 0
for i in range(0, len(move_sequence), 2):
    move = move_sequence[i:i + 2]
    idx = calculate_position(move)
    piece = 'x' if current_turn % 2 == 0 else 'o'
    chess_board = chess_board[:idx] + piece + chess_board[idx + 1:]
    current_turn += 1

result = []
for position in target_positions:
    lines = extract_directions(position, chess_board)
    result.append(','.join(lines))

final_answer = ','.join(result)
response = requests.get('http://183.175.12.27:8004/step_07', params={'ans': final_answer})
print(response.text)