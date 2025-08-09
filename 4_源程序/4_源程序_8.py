import requests as req


def calculate_index(position):
    return (ord(position[0]) - ord('a')) * 16 + ord(position[1]) - ord('a')


def generate_all_positions():
    positions = []
    for row in range(15):
        for col in range(16):
            positions.append(chr(row + 97) + chr(col + 97))
    return positions


def extract_patterns(position, board_state):
    patterns = ['', '', '', '']
    for offset in range(-4, 5):
        # 水平方向
        col = ord(position[1]) - ord('a') + offset
        if 0 <= col < 15:
            patterns[0] += board_state[(ord(position[0]) - ord('a')) * 16 + col]
        else:
            patterns[0] += ' '

        # 垂直方向
        row = ord(position[0]) - ord('a') + offset
        if 0 <= row < 15:
            patterns[2] += board_state[row * 16 + ord(position[1]) - ord('a')]
        else:
            patterns[2] += ' '

        # 左上到右下对角线
        row = ord(position[0]) - ord('a') + offset
        col = ord(position[1]) - ord('a') + offset
        if 0 <= row < 15 and 0 <= col < 15:
            patterns[1] += board_state[row * 16 + col]
        else:
            patterns[1] += ' '

        # 右上到左下对角线
        row = ord(position[0]) - ord('a') + offset
        col = ord(position[1]) - ord('a') - offset
        if 0 <= row < 15 and 0 <= col < 15:
            patterns[3] += board_state[row * 16 + col]
        else:
            patterns[3] += ' '
    return patterns


def determine_player_order(moves):
    return 'MO' if (len(moves) // 2) % 2 == 0 else 'OM'


def pattern_scoring_system():
    scoring = {
        ("CMMMM", "MCMMM", "MMCMM", "MMMCM", "MMMMC"): 10000,
        ("COOOO", "OOOOC"): 6000,
        (".CMMM.", ".MCMM.", ".MMCM.", ".MMMC."): 5000,
        ("COOO.", ".OOOC", ".OOCO.", ".OCOO."): 2500,
        ("OCMMM.", "OMCMM.", "OMMCM.", "OMMMC.", ".CMMMO", ".MCMMO", ".MMCMO", ".MMMCO"): 2000,
        (".MMC.", ".MCM.", ".CMM."): 400,
        (".OOC", "COO.", "MOOOC", "COOOM"): 400,
        (".MMCO", ".MCMO", ".CMMO", "OMMC.", "OMCM.", "OCMM.", "MOOC", "COOM"): 200,
        (".MC.", ".CM."): 50,
        ('.'): 20
    }
    return scoring


def find_optimal_move(move_sequence, scoring_system, position_map):
    # 初始化棋盘
    chess_board = '.' * 15 * 16

    # 确定玩家顺序
    player_sequence = determine_player_order(move_sequence)

    # 放置已有的棋子
    for i in range(0, len(move_sequence), 2):
        pos = calculate_index(move_sequence[i:i + 2])
        marker = player_sequence[0] if (i // 2) % 2 == 0 else player_sequence[1]
        chess_board = chess_board[:pos] + marker + chess_board[pos + 1:]

    # 寻找最优位置
    best_position = ''
    highest_score = 0

    for idx, cell in enumerate(chess_board):
        if cell == '.':
            # 试探性放置我方棋子
            temp_board = chess_board[:idx] + 'C' + chess_board[idx + 1:]
            current_pos = position_map[idx]

            # 评估四个方向的模式
            direction_patterns = ','.join(extract_patterns(current_pos, temp_board))

            # 计算总分
            current_score = 0
            for pattern_group, score in scoring_system.items():
                for pattern in pattern_group:
                    current_score += score * direction_patterns.count(pattern)

            # 更新最佳位置
            if current_score > highest_score:
                highest_score = current_score
                best_position = current_pos

    print(f"{best_position} {highest_score}")
    return best_position


def submit_answer(endpoint, solution):
    params = {'ans': solution[:-1]}
    response = req.get(endpoint, params=params)
    print(response.text)


# 主程序
api_url = "http://183.175.12.27:8004/step_08/"
response = req.get(api_url)
move_sequences = response.json()['questions']
scoring_rules = pattern_scoring_system()
position_lookup = generate_all_positions()

solution_string = ''
for sequence in move_sequences:
    solution_string += find_optimal_move(sequence, scoring_rules, position_lookup) + ','

submit_answer(api_url, solution_string)