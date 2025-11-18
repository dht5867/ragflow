import { ReactComponent as AssistantIcon } from '@/assets/svg/ai_bot.svg';
import { MessageType } from '@/constants/chat';
import { useSetModalState, useTranslate } from '@/hooks/common-hooks';
import { IChunk } from '@/interfaces/database/knowledge';
import { IReference, IReferenceChunk } from '@/interfaces/database/chat';
import classNames from 'classnames';
import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import { useLocation } from "react-router-dom";
import {
  useFetchDocumentInfosByIds,
  useFetchDocumentThumbnailsByIds,
} from '@/hooks/document-hooks';
import { IRegenerateMessage, IRemoveMessageById } from '@/hooks/logic-hooks';
import { IMessage } from '@/pages/chat/interface';
import MarkdownContent from '@/pages/chat/markdown-content';
import { getExtension, isImage } from '@/utils/document-util';
import {
  Avatar,
  Button,
  Flex,
  List,
  Space,
  Typography,
} from 'antd';
import askImage from '../../assets/ask.png'; // 引入本地图片
import FileIcon from '../file-icon';
import IndentedTreeModal from '../indented-tree/modal';
import NewDocumentLink from '../new-document-link';
import { useTheme } from '../theme-provider';
import { AssistantGroupButton, UserGroupButton } from './group-button';
import styles from './index.less';

const { Text } = Typography;

interface IProps extends Partial<IRemoveMessageById>, IRegenerateMessage {
  item: IMessage;
  reference: IReference;
  loading?: boolean;
  sendLoading?: boolean;
  visibleAvatar?: boolean;
  nickname?: string;
  avatar?: string;
  avatardialog?: string | null;
  clickDocumentButton?: (documentId: string, chunk: IReferenceChunk) => void;
  index: number;
  showLikeButton?: boolean;
  selectedSkill?: string;
  showLoudspeaker?: boolean;
}

const MessageItem = ({
  item,
  reference,
  loading = false,
  avatar,
  avatardialog,
  sendLoading = false,
  clickDocumentButton,
  index,
  removeMessageById,
  regenerateMessage,
  showLikeButton = true,
  selectedSkill,
  showLoudspeaker = true,
  visibleAvatar = true,
}: IProps) => {
  const { theme } = useTheme();
  const isAssistant = item.role === MessageType.Assistant;
  const isUser = item.role === MessageType.User;
  const { data: documentList, setDocumentIds } = useFetchDocumentInfosByIds();
  const { data: documentThumbnails, setDocumentIds: setIds } =
    useFetchDocumentThumbnailsByIds();
  const { visible, hideModal, showModal } = useSetModalState();
  const [clickedDocumentId, setClickedDocumentId] = useState('');
  const { t ,i18n} = useTranslate('chat');

  const referenceDocumentList = useMemo(() => {
    return reference?.doc_aggs ?? [];
  }, [reference?.doc_aggs]);

  const content = useMemo(() => {
    let text = item.content;
    if (text === '') {
      text = t('searching');
    }
    return loading ? text?.concat('~~2$$') : text;
  }, [item.content, loading, t]);

  const flowPath = useLocation();

  const skill = useMemo(() => {
    const isFlow = flowPath.pathname.includes("flow");
    console.log(flowPath.pathname)
    console.log(isFlow)
    if (isFlow){
      if ( i18n.language === 'zh' ){
        return '智能体'
      }
      if ( i18n.language === 'en' ){
        return 'Agents'
      }
      return 'Agents'
    }
  

    // 优化条件判断，减少冗余逻辑
    if (item.selectedSkill && item.selectedSkill.trim() !== '') {
      return item.selectedSkill;
    } else if (selectedSkill && selectedSkill.trim() !== '') {
      return selectedSkill;
    } else {
      if ( i18n.language === 'zh' ){
        return '自由对话'
      }
      if ( i18n.language === 'zh-TRADITIONAL' ){
        return '自由對話'
      }
      if ( i18n.language === 'en' ){
        return 'CHAT'
      }
      return 'CHAT'
    }
  }, [item.selectedSkill, selectedSkill]);

  const handleUserDocumentClick = useCallback(
    (id: string) => () => {
      setClickedDocumentId(id);
      showModal();
    },
    [showModal],
  );

  const handleRegenerateMessage = useCallback(() => {
    regenerateMessage?.(item);
  }, [regenerateMessage, item]);

  useEffect(() => {
    const ids = item?.doc_ids ?? [];
    if (ids.length) {
      setDocumentIds(ids);
      const documentIds = ids.filter((x) => !(x in documentThumbnails));
      if (documentIds.length) {
        setIds(documentIds);
      }
    }
  }, [item.doc_ids, setDocumentIds, setIds, documentThumbnails]);

  return (
    <div
      className={classNames(styles.messageItem, {
        [styles.messageItemLeft]: item.role === MessageType.Assistant,
        [styles.messageItemRight]: item.role === MessageType.User,
      })}
    >
      <section
        className={classNames(styles.messageItemSection, {
          [styles.messageItemSectionLeft]: item.role === MessageType.Assistant,
          [styles.messageItemSectionRight]: item.role === MessageType.User,
        })}
      >
        <div
          className={classNames(styles.messageItemContent, {
            [styles.messageItemContentReverse]: item.role === MessageType.User,
          })}
        >
           {visibleAvatar && 
           (item.role === MessageType.User ? (
            <Avatar size={40} src={askImage} />
          ) : avatardialog ? (
            <Avatar size={40} src={avatardialog} />
          ) : (
            <AssistantIcon />
          ))}
          <Flex vertical gap={8} flex={1}>
            <Space>
              {isAssistant ? (
                index !== 0 && (
                  <AssistantGroupButton
                    messageId={item.id}
                    content={item.content}
                    prompt={item.prompt}
                    showLikeButton={showLikeButton}
                    audioBinary={item.audio_binary}
                    showLoudspeaker={showLoudspeaker}
                  ></AssistantGroupButton>
                )
              ) : (
                <UserGroupButton
                  content={item.content}
                  messageId={item.id}
                  removeMessageById={removeMessageById}
                  regenerateMessage={
                    regenerateMessage && handleRegenerateMessage
                  }
                  sendLoading={sendLoading}
                ></UserGroupButton>
              )}
              {/* <b>{isAssistant ? '' : nickname}</b> */}
            </Space>
            
            <b>{isAssistant ? `【${skill || '自由对话'}】` : ''}</b>

            <div
              className={
                isAssistant
                  ? theme === 'dark'
                    ? styles.messageTextDark
                    : styles.messageText
                  : styles.messageUserText
              }
            >
              <MarkdownContent
                loading={loading}
                content={item.content}
                reference={reference}
                clickDocumentButton={clickDocumentButton}
              ></MarkdownContent>
            </div>
            {isAssistant &&referenceDocumentList.length > 0 &&(
                <List
                  bordered
                  dataSource={referenceDocumentList}
                  renderItem={(item) => {
                    return (
                      <List.Item>
                        <Flex gap={'small'} align="center">
                          <FileIcon
                            id={item.doc_id}
                            name={item.doc_name}
                          ></FileIcon>

                          <NewDocumentLink
                            documentId={item.doc_id}
                            documentName={item.doc_name}
                            prefix="document"
                          link={item.url}
                          >
                            {item.doc_name}
                          </NewDocumentLink>
                        </Flex>
                      </List.Item>
                    );
                  }}
                />
              )}
            {isUser && documentList.length > 0 && (
              <List
                bordered
                dataSource={documentList}
                renderItem={(item) => {
                  // TODO:
                  // const fileThumbnail =
                  //   documentThumbnails[item.id] || documentThumbnails[item.id];
                  const fileExtension = getExtension(item.name);
                  return (
                    <List.Item>
                      <Flex gap={'small'} align="center">
                        <FileIcon id={item.id} name={item.name}></FileIcon>

                        {isImage(fileExtension) ? (
                          <NewDocumentLink
                            documentId={item.id}
                            documentName={item.name}
                            prefix="document"
                          >
                            {item.name}
                          </NewDocumentLink>
                        ) : (
                          <Button
                            type={'text'}
                            onClick={handleUserDocumentClick(item.id)}
                          >
                            <Text
                              style={{ maxWidth: '40vw' }}
                              ellipsis={{ tooltip: item.name }}
                            >
                              {item.name}
                            </Text>
                          </Button>
                        )}
                      </Flex>
                    </List.Item>
                  );
                }}
              />
            )}
          </Flex>
        </div>
      </section>
      {visible && (
        <IndentedTreeModal
          visible={visible}
          hideModal={hideModal}
          documentId={clickedDocumentId}
        ></IndentedTreeModal>
      )}
    </div>
  );
};

export default memo(MessageItem);
