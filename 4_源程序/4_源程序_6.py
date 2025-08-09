import requests

def calculate_position(coordinate):
    col = ord(coordinate[0]) - ord('a')
    row = ord(coordinate[1]) - ord('a')
    return int(col * 16 + row)

api_endpoint = "http://183.175.12.27:8004/step_06/"
response = requests.get(api_endpoint)
move_sequence = response.json()['questions']

chess_board = '\n'.join(['.' * 15 for _ in range(15)])
move_count = 0
game_states = []

for move_index in range(0, len(move_sequence), 2):
    current_move = move_sequence[move_index:move_index + 2]
    position = calculate_position(current_move)
    board_chars = list(chess_board)
    piece = 'x' if move_count % 2 == 0 else 'o'
    board_chars[position] = piece
    chess_board = ''.join(board_chars)
    move_count += 1
    game_states.append(chess_board)

result_data = {
    'ans': ','.join(game_states)
}

final_response = requests.get(api_endpoint, params=result_data)
print(final_response.text)