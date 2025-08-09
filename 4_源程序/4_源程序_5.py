import requests

def calculate_mod_power(values):
    base, exponent, mod = values
    computed_value = 1
    while exponent:
        if exponent % 2:
            computed_value = (computed_value * base) % mod
        exponent //= 2
        base = (base * base) % mod
    return computed_value


def encode_string_to_integer(text):
    numeric_value = 0
    text_length = len(text)
    for index, char in enumerate(text):
        position = text_length - index - 1
        numeric_value += ord(char) * (256 ** position)
    return numeric_value


def decode_integer_to_string(numeric_value):
    char_values = []
    while numeric_value:
        char_values.append(numeric_value % 256)
        numeric_value //= 256

    char_values.reverse()
    return ''.join(chr(value) for value in char_values)

ENCRYPTION_EXPONENT = 65537
ENCRYPTION_MODULUS = 135261828916791946705313569652794581721330948863485438876915508683244111694485850733278569559191167660149469895899348939039437830613284874764820878002628686548956779897196112828969255650312573935871059275664474562666268163936821302832645284397530568872432109324825205567091066297960733513602409443790146687029

user_password = '1234'
password_numeric = encode_string_to_integer(user_password)
encrypted_password = calculate_mod_power([password_numeric, ENCRYPTION_EXPONENT, ENCRYPTION_MODULUS])
encrypted_hex = hex(encrypted_password)
print(f"加密后的密码: {encrypted_hex}")

request_params = {
    'user': 'name',
    'password': encrypted_hex
}

response = requests.get('http://183.175.12.27:8004/step_05/', params=request_params)

hex_prefix = '0x'
response_message = response.json()['message']
encrypted_response = int(hex_prefix + response_message, 16)
decrypted_value = calculate_mod_power([encrypted_response, ENCRYPTION_EXPONENT, ENCRYPTION_MODULUS])
decrypted_text = decode_integer_to_string(decrypted_value)

print(decrypted_text)
