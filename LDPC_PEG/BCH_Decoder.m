function corrected_msg = BCH_Decoder(n, k, m, msgs)
%     Description: Input m * n received codewords, recover those messages to m * k messages.
%     Input:
%         n: BCH Codeword length
%         k: BCH Message length
%         m: messages number(Inter-oligos length)    
%         megs: m * n received codeword messages(m datachunks).

%     Output:
%         decode_msg: m * k decoded messages(m datachunks), and 1-bit flag at the left most.
    
    % Generate BCH generator polynomial
    bchDecoder = comm.BCHDecoder(double(n), double(k));
    corrected_msg = zeros(m,k+1);
    flag = 0; % Flag for correctable or uncorrectable, flag = -1 if the sequence can not be corrected.
    for i = 1 : m 
%        if mod(i,1000) == 0
%            fprintf('BCH Decode Progress: %d/%d %.2f%%\r', i, m, 100*i/m);
%        end
        % Decode the message
        [de_msg, flag] = bchDecoder(msgs(i,:)');
        corrected_msg(i,1) = flag;
        corrected_msg(i,2:k+1) = de_msg';% Tranform vertical vertor to horizontal vector.
    end
end