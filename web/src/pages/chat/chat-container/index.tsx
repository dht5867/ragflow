import MessageItem from '@/components/message-item';
import { MessageType } from '@/constants/chat';
import { Flex, Spin } from 'antd';
import { useState } from 'react';
import {
  useCreateConversationBeforeUploadDocument,
  useGetFileIcon,
  useGetSendButtonDisabled,
  useSendButtonDisabled,
  useSendNextMessage,
} from '../hooks';
import { buildMessageItemReference } from '../utils';

import MessageInput from '@/components/message-input';
import PdfDrawer from '@/components/pdf-drawer';
import { useClickDrawer } from '@/components/pdf-drawer/hooks';
import {
  useFetchNextConversation,
  useFetchNextDialog,
  useGetChatSearchParams,
} from '@/hooks/chat-hooks';
import { useFetchUserInfo } from '@/hooks/user-setting-hooks';
import { buildMessageUuidWithRole } from '@/utils/chat';
import { memo } from 'react';
import AssistantIntro from './assistant-intro'; // 根据文件路径引入组件
import RenderIntro from './base-intro'; // 根据文件路径引入组件
import styles from './index.less';
import { useTranslate } from '@/hooks/common-hooks';

interface IProps {
  controller: AbortController;
}

const ChatContainer = ({ controller }: IProps) => {
  const { conversationId } = useGetChatSearchParams();
  const { data: conversation } = useFetchNextConversation();
  const { data: currentDialog } = useFetchNextDialog();
  const [selectedValue, setSelectedValue] = useState('');

  // 添加处理选择值的回调函数
  const handleSelect = (value: any) => {
    console.log('Selected value:', value);
    setSelectedValue(value); // 更新 selectedValue 状态
    // 在这里你可以处理选中的值，例如更新状态或进行其他操作
  };

  const {
    value,
    ref,
    loading,
    sendLoading,
    derivedMessages,
    handleInputChange,
    handlePressEnter,
    regenerateMessage,
    removeMessageById,
    stopOutputMessage,
  } = useSendNextMessage(controller, selectedValue);

  const { visible, hideModal, documentId, selectedChunk, clickDocumentButton } =
    useClickDrawer();
  const disabled = useGetSendButtonDisabled();
  const sendDisabled = useSendButtonDisabled(value);
  useGetFileIcon();
  const { data: userInfo } = useFetchUserInfo();
  const { createConversationBeforeUploadDocument } =
    useCreateConversationBeforeUploadDocument();

    const { t,i18n } = useTranslate('chat');
    
  return (
    <>
      <Flex flex={1} className={styles.chatContainer} vertical>
        <Flex flex={1} vertical className={styles.messageContainer}>
          {derivedMessages.length === 0 ? (
            // 当没有消息时显示 RenderIntro
            <RenderIntro selectedValue={selectedValue} language={i18n.language} />
          ) : (
            <div>
              <Spin spinning={loading}>
                {derivedMessages?.map((message, i) => {
                  return (
                    <MessageItem
                      loading={
                        message.role === MessageType.Assistant &&
                        sendLoading &&
                        derivedMessages.length - 1 === i
                      }
                      key={buildMessageUuidWithRole(message)}
                      item={message}
                      nickname={userInfo.nickname}
                      avatar={userInfo.avatar}
                      avatardialog={currentDialog.icon}
                      reference={buildMessageItemReference(
                        {
                          message: derivedMessages,
                          reference: conversation.reference,
                        },
                        message,
                      )}
                      clickDocumentButton={clickDocumentButton}
                      index={i}
                      removeMessageById={removeMessageById}
                      regenerateMessage={regenerateMessage}
                      sendLoading={sendLoading}
                      selectedSkill={selectedValue}
                    ></MessageItem>
                  );
                })}
              </Spin>
            </div>
          )}
          <div ref={ref} />
        </Flex>
        <AssistantIntro />

        <MessageInput
          disabled={disabled}
          sendDisabled={sendDisabled}
          sendLoading={sendLoading}
          value={value}
          onInputChange={handleInputChange}
          onPressEnter={handlePressEnter}
          conversationId={conversationId}
          createConversationBeforeUploadDocument={
            createConversationBeforeUploadDocument
          }
          stopOutputMessage={stopOutputMessage}
          onSelect={handleSelect} // 将 handleSelect 回调函数传递给 MessageInput
        ></MessageInput>
      </Flex>
      <PdfDrawer
        visible={visible}
        hideModal={hideModal}
        documentId={documentId}
        chunk={selectedChunk}
      ></PdfDrawer>
    </>
  );
};

export default memo(ChatContainer);
