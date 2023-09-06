import torch


class Model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict_masked_token(self, text):
        with torch.no_grad():
            text, input_label1, input_label2, input_distractor1, input_distractor2 = text['prompt'], text['label1'], text[
                'label2'], text['distractor1'], text['distractor2']

            # Encode the text
            encoded_input = self.tokenizer(text, return_tensors='pt')
            mask_token_index = torch.where(encoded_input['input_ids'] == self.tokenizer.mask_token_id)[1].item()
            # mask_token_index = (encoded_input.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

            # Get the label and distractor indices
            values = [input_label1, input_label2, input_distractor1, input_distractor2]
            label_index1 = self.tokenizer.encode(input_label1, add_special_tokens=False)[0]
            label_index2 = self.tokenizer.encode(input_label2, add_special_tokens=False)[0]
            distractor_index1 = self.tokenizer.encode(input_distractor1, add_special_tokens=False)[0]
            distractor_index2 = self.tokenizer.encode(input_distractor2, add_special_tokens=False)[0]

            # Predict the masked token
            output = self.model(**encoded_input)
            predictions = output.logits
            masked_token_predictions = predictions[0, mask_token_index]
            masked_token_predictions_softmax = torch.softmax(masked_token_predictions, dim=0)
            masked_token_predictions_softmax_label_distractor = masked_token_predictions_softmax[
                [label_index1, label_index2, distractor_index1, distractor_index2]]

            top_predicted_token = masked_token_predictions_softmax.argmax(axis=-1).item()
            decode_top_predicted_token = self.tokenizer.decode(top_predicted_token)
            # Get the predicted token index
            predicted_token_index = torch.argmax(masked_token_predictions_softmax_label_distractor).item()
            predicted_token = values[predicted_token_index]
            return predicted_token, decode_top_predicted_token


