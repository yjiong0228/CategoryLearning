import torch
from torch import nn, Tensor
from typing import Optional

class ExtendedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward_last(self, x: Tensor, mask: Tensor | None, state: Tensor):
        assert not self.norm_first
        x = self.norm1(x + self._sa_block_last_char(x, mask, state))
        x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block_last_char(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        state: Tensor
    ) -> Tensor:
        x = self.self_attn(x, state, state,
                           attn_mask=attn_mask,
                           key_padding_mask=None,
                           need_weights=False)[0]
        return self.dropout1(x)

    # self-attention block with attention weights output
    def _sa_block_with_attn(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        *,
        average_attn_weights=True
    ) -> tuple[Tensor, Tensor]:
        x, attn_map = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=average_attn_weights
        )
        return self.dropout1(x), attn_map


class ExtendedTransformerEncoder(nn.TransformerEncoder):
    def forward_last(self, x: Tensor, mask: Tensor | None, state: Tensor):
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, ExtendedTransformerEncoderLayer)
            state[i, :, -x.size(1):] = x
            x = layer.forward_last(x, mask, state[i])
        if self.norm is not None:
            x = self.norm(x)
        return x

    def get_attention_maps(self, x: Tensor, src_mask: Tensor | None = None, *, average_attn_weights=True):
        attention_maps = []
        for layer in self.layers:
            assert isinstance(layer, ExtendedTransformerEncoderLayer)
            assert layer.norm_first == False

            sa_x, attn_map = layer._sa_block_with_attn(
                x, src_mask, None, average_attn_weights=average_attn_weights)
            attention_maps.append(attn_map)
            x = layer.norm1(x + sa_x)
            del sa_x

            x = layer.norm2(x + layer._ff_block(x))

        return x, attention_maps


class ExtendedTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward_last(self, x: Tensor, memory: Tensor, mask: Tensor | None, state: Tensor):
        assert not self.norm_first
        x = self.norm1(x + self._sa_block_last_char(x, mask, state))
        x = self.norm2(x + self._mha_block(x, memory, None, None))
        x = self.norm3(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block_last_char(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        state: Tensor
    ) -> Tensor:
        x = self.self_attn(x, state, state,
                           attn_mask=attn_mask,
                           key_padding_mask=None,
                           need_weights=False)[0]
        return self.dropout1(x)

    # self-attention block with attention weights output
    def _sa_block_with_attn(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        *,
        average_attn_weights=True
    ) -> tuple[Tensor, Tensor]:
        x, attn_map = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=average_attn_weights
        )
        return self.dropout1(x), attn_map

    # multihead attention block with attention weights output
    def _mha_block_with_attn(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        *,
        average_attn_weights=True
    ) -> tuple[Tensor, Tensor]:
        x, attn_map = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=average_attn_weights
        )
        return self.dropout2(x), attn_map

    def _ff_block_with_neuron_value(self, x: Tensor) -> tuple[Tensor, Tensor]:
        neuron_value = self.activation(self.linear1(x))
        x = self.linear2(self.dropout(neuron_value))
        return self.dropout3(x), neuron_value


class ExtendedTransformerDecoder(nn.TransformerDecoder):
    def forward_last(self, x: Tensor, memory: Tensor, mask: Tensor | None, state: Tensor):
        for i, layer in enumerate(self.layers):
            assert isinstance(layer, ExtendedTransformerDecoderLayer)
            state[i, :, -x.size(1):] = x
            x = layer.forward_last(x, memory, mask, state[i])
        if self.norm is not None:
            x = self.norm(x)
        return x

    def get_attention_maps(self, x: Tensor, memory: Tensor, tgt_mask: Tensor, *, average_attn_weights=True):
        self_attention_maps = []
        multi_head_attention_maps = []
        for layer in self.layers:
            assert isinstance(layer, ExtendedTransformerDecoderLayer)
            assert layer.norm_first == False

            sa_x, sa_attn_map = layer._sa_block_with_attn(
                x, tgt_mask, None, average_attn_weights=average_attn_weights)
            self_attention_maps.append(sa_attn_map)
            x = layer.norm1(x + sa_x)
            del sa_x

            mha_x, mha_attn_map = layer._mha_block_with_attn(
                x, memory, None, None, average_attn_weights=average_attn_weights)
            multi_head_attention_maps.append(mha_attn_map)
            x = layer.norm2(x + mha_x)
            del mha_x

            x = layer.norm3(x + layer._ff_block(x))

        return x, self_attention_maps, multi_head_attention_maps

    def get_ff_neuron_maps(self, x: Tensor, memory: Tensor, tgt_mask: Tensor, neuron_index: Tensor):
        neuron_maps = []
        for layer, layer_neuron_index in zip(self.layers, neuron_index):
            assert isinstance(layer, ExtendedTransformerDecoderLayer)
            assert layer.norm_first == False

            x = layer.norm1(x + layer._sa_block(x, tgt_mask, None))
            x = layer.norm2(x + layer._mha_block(x, memory, None, None))
            ff_x, neuron_value = layer._ff_block_with_neuron_value(x)
            neuron_maps.append(neuron_value[:, :, layer_neuron_index])
            x = layer.norm3(x + ff_x)
            del ff_x, neuron_value

        return x, neuron_maps