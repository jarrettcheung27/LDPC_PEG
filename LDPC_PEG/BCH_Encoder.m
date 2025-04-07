function encoded_msgs = BCH_Encoder(n_bch, k_bch, m, msgs)
%     Description: Input k * 1 index messages, encode those messages into n * 1 index codeword.
%     Input:
%         n_bch: Codeword length
%         k_bch: Message length
%         m: messages number    
%         megs: k * m original messages(m datachunks).

%     Output:
%         encode_msg: n * m codewords(m datachunks).
    % fprintf('BCH decoding...')
    % Generate BCH generator polynomial
    bchEncoder = comm.BCHEncoder(double(n_bch), double(k_bch));
    
    % Encode messages
    encoded_msgs = zeros(m,n_bch);
    for i = 1 : m 
        % Encode the message
%        if mod(i,100) == 0
%            fprintf('Progress: %d/%d %.2f%%\r',i, m, 100*i/m);
%        end
        encoded_msgs(i,:) = (bchEncoder(msgs(i,:)'))';
    end
end